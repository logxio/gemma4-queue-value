"""Small numerical and causal-replay checks; no external data required."""
import unittest
import numpy as np
from openhands_queue_restart import boolean, estimates, schedule
from queue_restart_decision import one_schedule

class ReplayTests(unittest.TestCase):
    def test_boolean(self):
        for v in ('True','1','true'): self.assertTrue(boolean(v))
        for v in ('False','0','false'): self.assertFalse(boolean(v))
        with self.assertRaises(ValueError): boolean('')

    def test_pair_enumeration(self):
        rows=[[1,0],[0,1,1],[0,0,0,0]]
        n=np.array(list(map(len,rows)));k=np.array(list(map(sum,rows)))
        p,q,ratio,icc=estimates(n,k)
        failed=joint=0
        for r in rows:
            pairs=[(a,b) for i,a in enumerate(r) for j,b in enumerate(r) if i!=j]
            failed+=np.mean([not a for a,b in pairs])/len(rows)
            joint+=np.mean([(not a) and b for a,b in pairs])/len(rows)
        self.assertAlmostEqual(q,joint/failed)
        self.assertAlmostEqual(icc,1-ratio)

    def test_frozen_scheduler_equivalence(self):
        rng=np.random.default_rng(47);n=17
        s=rng.random((n,3))<.3;c=rng.integers(1,90,(n,3)).astype(float)
        cats=np.array(['a']*n);fresh={'a':.01,'__global__':.01}
        retry={(None,None,1):.02,(None,None,2):.003};edges=[20,60]
        for mode in ('plain_breadth','ranked_breadth','deferred_rate','adaptive'):
            generator=np.random.default_rng(92)
            perm=np.argsort(generator.random((n,3)),axis=1);order=generator.permutation(n)
            tr=schedule(np.take_along_axis(s,perm,axis=1),np.take_along_axis(c,perm,axis=1),
                np.zeros_like(s),cats,(fresh,retry,edges),mode,order)
            expected=[]
            one_schedule(s,c,cats,fresh,retry,edges,float('inf'),mode,np.random.default_rng(92),expected)
            np.testing.assert_array_equal(tr,np.asarray([(t[2],t[3],t[4]) for t in expected]))

    def test_depth_and_observed_failure_order(self):
        s=np.array([[False,True,False],[False,True,False]])
        c=np.array([[1,20,100],[90,4,100]])
        empty=np.array([[False,False,False],[True,False,False]])
        rates=({'__global__':1},{(None,None,1):1,(None,None,2):1},[10,50])
        for mode,expected in [('short_nonempty_first',[1,90,20,4]),('long_empty_first',[1,90,4,20]),('depth_first',[1,20,90,4])]:
            actual=schedule(s,c,empty,np.array(['a','a']),rates,mode,np.array([0,1]))
            self.assertEqual(actual[:,0].tolist(),expected)

if __name__=='__main__': unittest.main()
