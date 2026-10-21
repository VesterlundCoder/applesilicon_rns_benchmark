import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify',root/'verify_results.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class Verification(unittest.TestCase):
    def write_rns(self,path,ps,rows):
        path.write_bytes(struct.pack('<8sII',b'RNSR6I01',len(ps),6)+
            struct.pack(f'<{len(ps)}I',*ps)+struct.pack(f'<{36*len(ps)}I',*(x for row in rows for x in row)))

    def test_signed_boundaries(self):
        for ps in ([5,7],[2,5,7]):
            M=1
            for p in ps:M*=p
            for x in range(M):
                got,modulus=v.crt_centered(ps,[[x%p]*36 for p in ps])
                self.assertEqual(modulus,M)
                self.assertEqual(got,[x if 2*x<M else x-M]*36)

    def test_invalid_moduli_and_residues(self):
        for ps,rows in [([],[]),([101,101],[[1]*36]*2),([105],[[1]*36]),
                         ([101],[[101]*36]),([101],[[1]*35]),([2**31+11],[[1]*36])]:
            with self.assertRaises(ValueError):v.crt_centered(ps,rows)

    def test_dump_capacity_is_separate_from_residue_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'dump.rns';self.write_rns(p,[101],[[1]*36])
            row=v.verify_rns(p,[102]*36)
            self.assertFalse(row['capacity_sufficient'])
            self.assertFalse(row['correct'])

    def test_valid_exact_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'dump.rns';self.write_rns(p,[101,103],[[(-7)%p]*36 for p in [101,103]])
            self.assertTrue(v.verify_rns(p,[-7]*36)['correct'])

    def test_malformed_binary_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'dump.rns';self.write_rns(p,[],[])
            with self.assertRaises(ValueError):v.load_rns(p)
            p.write_bytes(b'truncated')
            with self.assertRaises(ValueError):v.load_rns(p)

    def test_nonfinite_float_dump_fails_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);dump=tmp/'float.bin';ref=tmp/'reference.json'
            dump.write_bytes(struct.pack('<8sIq36d',b'F64R6I01',6,0,float('nan'),*([1.0]*35)))
            ref.write_text(json.dumps({'entries_row_major':['1']*36}))
            result=subprocess.run([sys.executable,str(root/'verify_results.py'),'--reference',str(ref),'--f64',str(dump)],capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertFalse(json.loads(result.stdout)['all_finite'])

if __name__=='__main__':unittest.main()
