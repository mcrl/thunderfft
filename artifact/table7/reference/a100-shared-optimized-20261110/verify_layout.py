#!/usr/bin/env python3
"""Exhaustive physical-address/bank proof for fixed shared FFT layouts."""
import json

def address(t, i, c, vector):
    parity = ((i >> 1) ^ (i >> 3) ^ (i >> 5)) & 1
    packed = (i & 1) | ((i >> 2) << 1)
    if vector:
        return (packed * 2 + ((t >> 2) & 1)) * 32 + (t & 3) * 8 + c * 4 + parity * 2 + (t >> 3)
    return ((t >> 3) * 32 + packed) * 32 + (t & 7) * 4 + c * 2 + parity

results = {}
for vector in (False, True):
    all_addresses = [address(t,i,c,vector) for t in range(16) for i in range(64) for c in range(2)]
    assert len(set(all_addresses)) == 2048
    assert set(all_addresses) == set(range(2048))
    checked_transactions = 0
    for stage in range(3):
        stride = 4 ** stage
        for bf in range(16):
            base = (bf // stride) * 4 * stride + bf % stride
            for ki in range(2):
                if vector:
                    for half in (0, 16):
                        words=[]
                        for lane in range(half, half+16):
                            t=lane//4; i=base+stride*(2*(lane&1)+ki); c=(lane%4)//2
                            a=address(t,i,c,True); b=address(t+8,i,c,True)
                            assert a%2==0 and b==a+1
                            words.extend((a,b))
                        assert len(set(words))==32
                        assert len({a%32 for a in words})==32
                        checked_transactions += 1
                else:
                    for upper in (0,8):
                        words=[address(lane//4+upper,base+stride*(2*(lane&1)+ki),(lane%4)//2,False) for lane in range(32)]
                        assert len(set(words))==32
                        assert len({a%32 for a in words})==32
                        checked_transactions += 1
    results['vector2' if vector else 'scalar'] = dict(bijective_words=2048, shared_bytes=8192, bank_unique_transactions=checked_transactions, maximum_bank_conflict=1)
print(json.dumps(results,indent=2))
