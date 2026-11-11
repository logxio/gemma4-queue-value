#!/usr/bin/env python3
"""CPU-only audit of existing Gemma traces; exports aggregates and redacted tables."""
import argparse, collections, csv, hashlib, itertools, json, math, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ['edit_errors', 'schema_errors', 'other_edit_errors', 'timeout', 'context_window', 'early_loop', 'normal_failure']

def ratio(a,b): return a/b if b else None

def ci(xs):
    xs=sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs: return [None,None]
    out=[]
    for p in (.025,.975):
        h=(len(xs)-1)*p; lo=math.floor(h); hi=math.ceil(h)
        out.append(xs[lo]+(h-lo)*(xs[hi]-xs[lo]))
    return out

def obs(step):
    c=step.get('observation',{}).get('content',{})
    if isinstance(c,str):
        try: return json.loads(c)
        except (ValueError,TypeError): return {}
    return c if isinstance(c,dict) else {}

def fisher(a,b,c,d):
    n=a+b+c+d; k=a+c; m=a+b
    def pr(x): return math.comb(k,x)*math.comb(n-k,m-x)/math.comb(n,m)
    p=pr(a)
    return min(1.,sum(pr(x) for x in range(max(0,m-(n-k)),min(k,m)+1) if pr(x)<=p*(1+1e-10)))

def matched_sign(diffs):
    # Exact sign test on per-task difference in success fractions; ties removed.
    pos=sum(d>0 for d in diffs); neg=sum(d<0 for d in diffs); n=pos+neg
    return {'positive_tasks':pos,'negative_tasks':neg,'ties':len(diffs)-n,
            'two_sided_sign_p':min(1.,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else None}

def rates(groups):
    rows=[r for g in groups for r in g]; successes=sum(r['resolved'] for r in rows)
    # Ordered distinct repeat pairs: no ordering assumptions; also report chronological separately.
    pairs=[(a,b) for g in groups for a in g for b in g if a['round']!=b['round'] and not a['resolved']]
    p=ratio(successes,len(rows)); q=ratio(sum(b['resolved'] for a,b in pairs),len(pairs))
    return {'p':p,'q':q,'q_over_p':ratio(q,p) if p else None,'successes':successes,'attempts':len(rows),
            'restart_successes':sum(b['resolved'] for a,b in pairs),'restart_pairs':len(pairs)}

def mode_stats(groups,mode):
    rows=[r for g in groups for r in g]; exposed=[r for r in rows if r[mode]>0]; unexposed=[r for r in rows if not r[mode]]
    a=sum(r['resolved'] for r in exposed); b=len(exposed)-a; c=sum(r['resolved'] for r in unexposed); d=len(unexposed)-c
    diffs=[]
    for g in groups:
        e=[r['resolved'] for r in g if r[mode]]; u=[r['resolved'] for r in g if not r[mode]]
        if e and u: diffs.append(statistics.mean(e)-statistics.mean(u))
    pairs=[(x,y) for g in groups for x in g for y in g if x['round']!=y['round'] and not x['resolved'] and x[mode]]
    nextpairs=[(x,y) for x,y in pairs if y['round']==x['round']+1]
    return {'exposed_successes':a,'exposed_failures':b,'unexposed_successes':c,'unexposed_failures':d,
            'exposed_success_rate':ratio(a,a+b),'unexposed_success_rate':ratio(c,c+d),
            'risk_difference':ratio(a,a+b)-ratio(c,c+d) if (a+b)*(c+d) else None,
            'attempt_fisher_p_descriptive_only':fisher(a,b,c,d),
            'within_task_exposure_discordant_tasks':len(diffs),'within_task_mean_risk_difference':statistics.mean(diffs) if diffs else None,
            'within_task_sign_test':matched_sign(diffs),
            'restart_pairs':len(pairs),'restart_successes':sum(y['resolved'] for x,y in pairs),
            'restart_rate':ratio(sum(y['resolved'] for x,y in pairs),len(pairs)),
            'chronological_next_pairs':len(nextpairs),'chronological_next_successes':sum(y['resolved'] for x,y in nextpairs),
            'chronological_next_rate':ratio(sum(y['resolved'] for x,y in nextpairs),len(nextpairs))}

def replay(groups,policy,late_recovery=0.,rng=None):
    # Each resampled task is a separate queue identity. Broad-first round order, skip resolved identities.
    elapsed=0.; solved=set(); fresh_started=set(); fresh_completed=set(); completed=started=stopped=0
    for rd in (1,2,3):
        for identity,g in enumerate(groups):
            if identity in solved: continue
            row=next(r for r in g if r['round']==rd)
            duration=min(row['seconds'],600.); success=bool(row['resolved'] and row['seconds']<=600.)
            trigger=row.get(policy) if policy.startswith('fuse_') else None
            if trigger is not None and trigger<duration:
                duration=trigger; success=False; stopped+=1
            if policy=='assumed_60min' and row['timed_out']:
                duration=3600.; success=(rng.random()<late_recovery) if rng else False
            started+=1; fresh_started.add(identity)
            if elapsed+duration>43200:
                return {'solved':len(solved),'completed_attempts':completed,'started_attempts':started,'seconds_used':43200.,'breaker_stops':stopped,'distinct_issues_started':len(fresh_started),'distinct_issues_completed':len(fresh_completed)}
            elapsed+=duration; completed+=1; fresh_completed.add(identity)
            if success: solved.add(identity)
    return {'solved':len(solved),'completed_attempts':completed,'started_attempts':started,'seconds_used':elapsed,'breaker_stops':stopped,'distinct_issues_started':len(fresh_started),'distinct_issues_completed':len(fresh_completed)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=1000);ap.add_argument('--seed',type=int,default=20261109);args=ap.parse_args()
    base=ROOT/'evidence/results/gemma4_sandbox_and_schema_audit'
    refpath=ROOT/'evidence/gemma_a100/reference_patch_check_129.json'; rowpath=ROOT/'evidence/gemma_a100/full76_official.jsonl'
    legacypath=ROOT/'evidence/gemma_a100/legacy_summary.json'
    legacyattemptpath=legacypath.with_name('attempts.json')
    scriptpath=ROOT/'analysis/analyze_stability.py'
    reference=json.loads(refpath.read_text()); legacy=json.loads(legacypath.read_text())
    rows=[json.loads(s) for s in rowpath.read_text().splitlines() if s]
    paths={}; events=[]; missing_time=0; trace_hashes=[]
    for r in rows:
        paths[(r['task'],r['round'])]=list((ROOT/'.ai/gpu/runs/full76/results'/f"r{r['round']}").glob(f"*/traces/trace_{r['task']}.json"))
        assert len(paths[(r['task'],r['round'])])==1
        trace_bytes=paths[(r['task'],r['round'])][0].read_bytes()
        trace_hashes.append({'task':r['task'],'round':r['round'],'sha256':hashlib.sha256(trace_bytes).hexdigest()})
        trace=json.loads(trace_bytes)
        calls=[c for step in trace['steps'] for c in step.get('tool_calls',[])]
        calls.sort(key=lambda c:c.get('extra',{}).get('timestamp',0.))
        exact=[c.get('function_name','')+'|'+json.dumps(c.get('arguments',{}),sort_keys=True) for c in calls]
        near=[c.get('function_name','')+'|'+json.dumps(c.get('arguments',{}),sort_keys=True)[:80] for c in calls]
        exact_runs=[len(list(v)) for _,v in itertools.groupby(exact)]
        near_runs=[len(list(v)) for _,v in itertools.groupby(near)]
        r['longest_exact_repeat']=max(exact_runs,default=0);r['exact_loop_flag']=r['longest_exact_repeat']>=5
        r['longest_near_repeat']=max(near_runs,default=0);r['near_loop_flag']=r['longest_near_repeat']>=5
        assert sum(n for n in near_runs if n>=5)==r['loop_calls']
        r.update(edit_errors=0,schema_errors=0,other_edit_errors=0,schema_observed_latency_seconds=0.,schema_preceding_event_interval_seconds=0.)
        streak={'edit':0,'schema':0};r.update({f'fuse_{m}_{k}':None for m in streak for k in (2,3,5)})
        prev=None
        for s in trace['steps']:
            t=s.get('extra',{}).get('elapsed_s'); calls=s.get('tool_calls',[])
            isedit=any(c['function_name']=='edit_file' for c in calls); o=obs(s)
            error=isedit and bool(o.get('error') or o.get('status')=='error')
            schema=error and 'mandatory input parameters' in str(o)
            if error:
                r['edit_errors']+=1;r['schema_errors']+=int(schema);r['other_edit_errors']+=int(not schema)
                ot=s.get('observation',{}).get('extra',{}).get('elapsed_s')
                ct=next((c.get('extra',{}).get('elapsed_s') for c in calls if c['function_name']=='edit_file'),None)
                latency=ot-ct if ot is not None and ct is not None and ot>=ct else None
                interval=t-prev if t is not None and prev is not None and t>=prev else None
                if schema:
                    if latency is None:missing_time+=1
                    else:r['schema_observed_latency_seconds']+=latency
                    if interval is not None:r['schema_preceding_event_interval_seconds']+=interval
                events.append({'task':r['task'],'round':r['round'],'step_id':s['step_id'],'schema_error':bool(schema),'call_elapsed_seconds':ct,'observation_elapsed_seconds':ot,'tool_observation_latency_seconds':latency,'preceding_event_interval_seconds':interval})
            if calls:
                for m,v in [('edit',error),('schema',schema)]:
                    streak[m]=streak[m]+1 if v else 0
                    for k in (2,3,5):
                        key=f'fuse_{m}_{k}'
                        if streak[m]>=k and r[key] is None:r[key]=s.get('observation',{}).get('extra',{}).get('elapsed_s',t)
            if t is not None:prev=t
        r.update(timeout=r['timed_out'],context_window=r['failure_kind']=='context_window',early_loop=r['loop_flag_by_20_calls'],normal_failure=r['failure_kind']=='none')
    bytask=collections.defaultdict(list)
    for r in rows:bytask[r['task']].append(r)
    groups=[sorted(bytask[k],key=lambda r:r['round']) for k in sorted(bytask)]
    assert len(groups)==76 and len(rows)==228 and all(len(g)==3 for g in groups)
    assert {r['task'] for r in reference['tasks'] if r['reference_patch_resolved']}==set(bytask)
    audit={}
    for repo in sorted({r['task'].split('_')[0] for r in reference['tasks']}):
        rr=[r for r in reference['tasks'] if r['task'].startswith(repo+'_')];fails=[r for r in rr if not r['reference_patch_resolved']]
        audit[repo]={'total':len(rr),'reference_pass':len(rr)-len(fails),'reference_fail':len(fails),'pass_fraction':ratio(len(rr)-len(fails),len(rr)),
                     'failed_with_missing_module':sum(bool(r['missing_modules']) for r in fails),'failed_without_listed_missing_module':sum(not r['missing_modules'] for r in fails),
                     'missing_module_counts':dict(collections.Counter(m for r in fails for m in r['missing_modules']))}
    paired_groups=[g for g in groups if any(r['resolved'] for r in g) and any(not r['resolved'] and r['failure_kind']=='none' for r in g)]
    paired_tasks=[];paired_rows=[]
    for g in paired_groups:
        successes=[r for r in g if r['resolved']];failures=[r for r in g if not r['resolved'] and r['failure_kind']=='none']
        paired_tasks.append({'task':g[0]['task'],'success_rounds':[r['round'] for r in successes],'normal_failed_rounds':[r['round'] for r in failures],
            'excluded_execution_error_rounds':[r['round'] for r in g if not r['resolved'] and r['failure_kind']!='none'],
            **{f'{mode}_{outcome}_flagged_attempts':sum(r[mode+'_loop_flag'] for r in rs) for mode in ('exact','near') for outcome,rs in [('success',successes),('failure',failures)]}})
        for r in g:paired_rows.append({**r,'paired_eligible_attempt':r['resolved'] or r['failure_kind']=='none'})
    paired_report={'selection':'Task has at least one resolved attempt and one unresolved attempt with failure_kind == none. Execution-error failures in those tasks are excluded from paired comparison.',
        'exact_loop_definition':'At least 5 consecutive tool calls with identical function name and complete json.dumps(arguments, sort_keys=True), ordered by call timestamp, across all authors.',
        'near_loop_definition':'Same as legacy loop_calls: function name plus first 80 characters of sorted JSON arguments; this can group different full arguments.',
        'tasks':len(paired_tasks),'records':paired_tasks,
        'success_attempts':sum(len(r['success_rounds']) for r in paired_tasks),'normal_failure_attempts':sum(len(r['normal_failed_rounds']) for r in paired_tasks),
        'requested_13_9_0':{'tasks':13,'exact_failure_flagged_attempts':9,'exact_success_flagged_attempts':0},
        'recomputed':{f'{m}_{o}_flagged_attempts':sum(r[f'{m}_{o}_flagged_attempts'] for r in paired_tasks) for m in ('exact','near') for o in ('success','failure')},
        'near_only_success_records':[{'task':r['task'],'round':r['round'],'longest_exact_repeat':r['longest_exact_repeat'],'longest_near_repeat':r['longest_near_repeat']} for r in paired_rows if r['resolved'] and r['near_loop_flag'] and not r['exact_loop_flag']]}
    paired_report['exact_task_matched_sign_test']=matched_sign([int(r['exact_success_flagged_attempts']>0)-int(r['exact_failure_flagged_attempts']>0) for r in paired_tasks])
    assert paired_report['tasks']==13 and paired_report['recomputed']['exact_failure_flagged_attempts']==9 and paired_report['recomputed']['exact_success_flagged_attempts']==0
    stats={m:mode_stats(groups,m) for m in MODES}; mainrates=rates(groups)
    legacyattempts=json.loads(legacyattemptpath.read_text())
    legacyindex={(r['task'],r['round']):r for r in legacyattempts}
    assert all(r['edit_errors']==legacyindex[(r['task'],r['round'])]['edit_errors'] and r['schema_errors']==legacyindex[(r['task'],r['round'])]['schema_errors'] for r in rows)
    bootrates=collections.defaultdict(list);bootstats={m:collections.defaultdict(list) for m in MODES}; rng=random.Random(args.seed)
    policies=['cap_600','fuse_schema_2','fuse_schema_3','fuse_schema_5','fuse_edit_3']
    queue={p:[] for p in policies};sensitivity={str(p):[] for p in (0.,.1,.25,.5,1.)};queue_deltas={p:[] for p in policies[1:]}; sensdelta={p:[] for p in sensitivity}
    for _ in range(args.bootstrap):
        sample=[rng.choice(groups) for _ in groups];rng.shuffle(sample)
        rr=rates(sample)
        for k in ('p','q','q_over_p'):bootrates[k].append(rr[k])
        for m in MODES:
            ss=mode_stats(sample,m)
            for k in ('risk_difference','within_task_mean_risk_difference','restart_rate','chronological_next_rate'):bootstats[m][k].append(ss[k])
        runs={p:replay(sample,p) for p in policies}
        for p in policies:queue[p].append(runs[p])
        for p in policies[1:]:queue_deltas[p].append(runs[p]['solved']-runs['cap_600']['solved'])
        for p in sensitivity:
            ss=replay(sample,'assumed_60min',float(p),rng);sensitivity[p].append(ss);sensdelta[p].append(runs['fuse_schema_3']['solved']-ss['solved'])
    for m in MODES:stats[m]['task_bootstrap_95ci']={k:ci(v) for k,v in bootstats[m].items()}
    for m in ('normal_failure','timeout','context_window'):
        for k in ('attempt_fisher_p_descriptive_only','within_task_sign_test','within_task_mean_risk_difference','risk_difference','exposed_success_rate','unexposed_success_rate'):
            stats[m][k]=None
        stats[m]['task_bootstrap_95ci']={k:v for k,v in stats[m]['task_bootstrap_95ci'].items() if k in ('restart_rate','chronological_next_rate')}
        stats[m]['association_not_tested_reason']='Failure status is outcome-related; retain decomposition and conditional restart estimates only.'
    def summarize_runs(rs):return {k:{'mean':statistics.mean(r[k] for r in rs),'task_bootstrap_95ci':ci([r[k] for r in rs])} for k in rs[0]}
    def summarize_delta(xs):return {'mean':statistics.mean(xs),'task_bootstrap_95ci':ci(xs)}
    nsuccess=mainrates['successes']; padded=groups+[[{'resolved':False,'round':i} for i in (1,2,3)] for _ in range(53)]
    discrepancies={m:{'recomputed_failed_attempts':stats[m]['exposed_failures'],'legacy_failed_attempts':legacy['blocks'][{'schema_errors':'edit_schema_errors'}.get(m,m)]['attempts']} for m in MODES if m!='other_edit_errors'}
    assert all(v['recomputed_failed_attempts']==v['legacy_failed_attempts'] for v in discrepancies.values())
    result={'schema_version':1,'seed':args.seed,'bootstrap_replicates':args.bootstrap,'percentile_definition':'Linear interpolation: h=(n-1)*p, quantile=x[floor(h)]+(h-floor(h))*(x[ceil(h)]-x[floor(h)]), p=0.025 and 0.975; NumPy percentile method=linear.','bootstrap_unit':'task: preserve all three rounds; resample 76 task clusters with replacement',
      'sources':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [refpath,rowpath,legacypath,legacyattemptpath,scriptpath,ROOT/'analysis/gemma_case_study.py',ROOT/'analysis/gemma_rollouts.py',Path(__file__)]},
      'trace_manifest':trace_hashes,
      'same_task_success_normal_failure_pairing':paired_report,
      'reference_environment':reference['environment'],'reference_audit':audit,
      'per_repo_observed_rates':{repo:rates([g for g in groups if g[0]['repo']==repo]) for repo in sorted({r['repo'] for r in rows})},
      'reference_interpretation':'Reference-patch pass/fail in the recorded environment, not proof of intrinsic task solvability or official-server behavior. Missing-module attribution is restricted to listed modules.',
      'observed':{'tasks':len(groups),'attempts':len(rows),'successes':nsuccess,'failures':len(rows)-nsuccess,'union_solved':sum(any(r['resolved'] for r in g) for g in groups),
        'rates':mainrates,'rate_bootstrap_95ci':{k:ci(v) for k,v in bootrates.items()},'error_modalities':stats,'legacy_crosscheck':discrepancies,'successful_edit_error_claim_audit':{'requested_claim':19,'raw_recomputed':stats['edit_errors']['exposed_successes'],'legacy_attempts_recomputed':sum(r['resolved'] and r['edit_errors']>0 for r in legacyattempts),'per_attempt_error_count_discrepancies':[],'two_difference_records':None,'identification_reason':'All 228 edit and schema counts agree exactly with legacy attempts.json. Legacy summary only reports failed attempts; no 19-success definition or row set was supplied.','successful_exposed_records':[{'task':r['task'],'round':r['round'],'edit_errors':r['edit_errors'],'schema_errors':r['schema_errors']} for r in rows if r['resolved'] and r['edit_errors']]} },
      'denominator_distortion':{'observed_judgable':mainrates,'zero_imputed_53_unrun_tasks':rates(padded),
          'zero_imputation_is_counterfactual':True,'p_multiplicative_attenuation':76/129,
          'definition':'p = successes/attempts. q = successes among all ordered distinct attempt pairs whose first attempt failed. No outcomes observed for 53 excluded tasks; zero-imputation illustrates measurement distortion only.'},
      'ghost_costs':{'schema_rejected_calls':sum(r['schema_errors'] for r in rows),'attempts_with_schema_error':sum(bool(r['schema_errors']) for r in rows),
        'observed_schema_call_to_observation_latency_seconds':sum(r['schema_observed_latency_seconds'] for r in rows),'missing_latency_events':missing_time,
        'preceding_event_intervals_seconds':sum(r['schema_preceding_event_interval_seconds'] for r in rows),
        'causal_schema_wallclock_cost_seconds':None,'causal_cost_identification_reason':'Per-call dispatch and observation timestamps identify response latency; preceding event gaps also include generation, scheduling and intervening work. No no-error counterfactual or generation-start timestamps identify total causal error cost.',
        'uncounted_schema_calls_direct_counter_proof':None,'all_llm_calls':sum(r['llm_calls'] for r in rows),'all_counted_tool_calls':sum(r['tool_calls'] for r in rows),
        'schema_calls_by_outcome':{str(success):sum(r['schema_errors'] for r in rows if r['resolved']==success) for success in (False,True)},
        'counter_identification_reason':'Traces expose rejected calls and final counted tool calls but no per-call budget counter transitions. Rejected-call count is observed; counter-exclusion mechanism needs harness source corroboration.',
        'llm_minus_tool_calls':sum(r['llm_calls']-r['tool_calls'] for r in rows),'llm_minus_tool_interpretation':'Includes successful delegation and text-only calls; not a ghost-call count.'},
      'cases':[{k:r[k] for k in ['task','round','resolved','seconds','tool_calls','llm_calls','edit_errors','schema_errors','schema_observed_latency_seconds','schema_preceding_event_interval_seconds','fuse_schema_3']} for r in sorted(rows,key=lambda r:r['schema_errors'],reverse=True)[:8]],
      'queue':{'budget_seconds':43200,'policy':'Broad-first rounds 1,2,3, skipping solved tasks; bootstrap resamples and permutes task order. Trigger is kth consecutive error-bearing tool step, reset by any nonmatching tool step; text/system steps do not reset. Triggered attempt treated as unsuccessful; final patch at earlier trigger not tested.',
        'measured_support':'Observed attempts were capped at 600 seconds with termination overhead. Replay truncates overhead to 600. No 60-minute trajectories are observed.',
        'observed_task_order_point_replay':{p:replay([sorted(g,key=lambda r:r['round']) for g in bytask.values()],p) for p in policies},
        'replay_600_and_breakers':{p:summarize_runs(rs) for p,rs in queue.items()},'paired_breaker_minus_600_solved':{p:summarize_delta(v) for p,v in queue_deltas.items()},
        'paired_breaker_minus_600_distinct_issues':{p:{k:summarize_delta([x[k]-y[k] for x,y in zip(queue[p],queue['cap_600'])]) for k in ('distinct_issues_started','distinct_issues_completed')} for p in policies[1:]},
        'observed_trigger_suffixes':{p:{'triggered_attempts':sum(r.get(p) is not None and r[p]<min(r['seconds'],600.) for r in rows),'triggered_eventual_successes':sum(r['resolved'] and r.get(p) is not None and r[p]<min(r['seconds'],600.) for r in rows),'removed_observed_suffix_seconds':sum(min(r['seconds'],600.)-r[p] for r in rows if r.get(p) is not None and r[p]<min(r['seconds'],600.))} for p in policies[1:]},
        'breaker_is_simulation':'Trigger times and removed suffixes observed; changed outcomes and extra queued completions are replay assumptions, not measured online gains.',
        'assumed_60min_sensitivity':{p:{'assumed_probability_of_late_recovery':float(p),'queue':summarize_runs(rs),'paired_schema3_minus_60_solved':summarize_delta(sensdelta[p]),'paired_schema3_minus_60_distinct_issues':{k:summarize_delta([x[k]-y[k] for x,y in zip(queue['fuse_schema_3'],rs)]) for k in ('distinct_issues_started','distinct_issues_completed')}} for p,rs in sensitivity.items()},
        '60min_assumptions':'Only timed_out attempts extend to exactly 3600 seconds; all other durations/outcomes unchanged. Late success is Bernoulli at 3600 seconds with stated probability. No early late-success timing modeled; no recovery effects inferred from observed data.',
        'actual_60min_counterfactual':None,'actual_60min_identification_reason':'No outcomes or event history beyond observed 600-second cap.'},
      'redacted_table_contract':{'attempts':'All 228 rows preserve task clusters, outcomes, error counts, runtime, failure and loop flags, observed error latencies, and all evaluated breaker triggers; enough to recompute p/q, exposure associations, conditional restart and queue simulation without private traces.','events':'Edit-error event times only; no code, prompts, arguments or observation text. Sums reproduce observed latency and event-gap aggregates.','reference':'129 reference outcomes and module labels, enough to recompute repository audit.','draws':'1000 paired task-bootstrap queue draws per policy, enough to independently summarize queue means, percentile intervals and paired differences. Statistical rate and exposure intervals require rerunning seeded task bootstrap on attempts.','trace_hashes':'Per task and round SHA256 manifest validates private raw sources without exporting them.'},
      'inference_notes':['Fisher p-values ignore within-task dependence and are descriptive only; task bootstrap and within-task sign test handle task clustering.','Exposure is post-treatment and may reflect task difficulty or trajectory duration; association is not causal.','Multiple error modalities overlap; no multiple-testing-adjusted claims made.','Chronological next-round rates and unordered repeat-pair rates are distinct estimands.']}
    base.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['task','repo','round','resolved','seconds','tool_calls','llm_calls','timed_out','failure_kind','loop_flag_by_20_calls','loop_calls','longest_exact_repeat','exact_loop_flag','longest_near_repeat','near_loop_flag','timeout','context_window','early_loop','normal_failure','edit_errors','schema_errors','other_edit_errors','schema_observed_latency_seconds','schema_preceding_event_interval_seconds','fuse_schema_2','fuse_schema_3','fuse_schema_5','fuse_edit_3']
    for suffix,data,keys in [('_attempts.csv',rows,fields),('_events.csv',events,list(events[0])),('_reference.csv',reference['tasks'],['task','reference_patch_resolved','test_exit_code','missing_modules']),('_paired_attempts.csv',paired_rows,fields+['paired_eligible_attempt'])]:
        with Path(str(base)+suffix).open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(data)
    draws=[]
    for policy,rs in list(queue.items())+[(f'assumed_60min_recovery_{p}',rs) for p,rs in sensitivity.items()]:
        for i,row in enumerate(rs):draws.append({'bootstrap_draw':i,'policy':policy,**row})
    with Path(str(base)+'_queue_draws.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(draws[0]));w.writeheader();w.writerows(draws)
    print(json.dumps({'result':str(base.with_suffix('.json')),'rates':mainrates,'schema_calls':result['ghost_costs']['schema_rejected_calls'],'queue_delta':result['queue']['paired_breaker_minus_600_solved']},indent=2))

if __name__=='__main__':main()
