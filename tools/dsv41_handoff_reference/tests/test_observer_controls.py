import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bin'))
from observer_controls import armed,update


class ObserverControlTests(unittest.TestCase):
    def test_arm_and_clear_both_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'p';d=root/'d';p.mkdir();d.mkdir()
            cfg=root/'controls.json';cfg.write_text(json.dumps({role:dict(root=str(path),folder='audit') for role,path in [('producer',p),('consumer',d)]}))
            with armed(cfg,'abc123',[1,2],'identity'):
                self.assertEqual(json.loads((p/'reference-control.json').read_text())['action'],'observe_source')
                self.assertEqual(json.loads((d/'reference-control.json').read_text())['action'],'observe')
            self.assertFalse((p/'reference-control.json').exists());self.assertFalse((d/'reference-control.json').exists())

    def test_second_role_conflict_rolls_back_only_owned_first_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'p';d=root/'d';p.mkdir();d.mkdir()
            cfg=root/'controls.json';cfg.write_text(json.dumps({role:dict(root=str(path),folder='audit') for role,path in [('producer',p),('consumer',d)]}))
            other=d/'reference-control.json';other.write_text(json.dumps(dict(key='another')))
            with self.assertRaises(Exception):
                with armed(cfg,'abc123',[1,2],'identity'):self.fail('Must not enter')
            self.assertFalse((p/'reference-control.json').exists());self.assertEqual(json.loads(other.read_text())['key'],'another')

    def test_exception_disarms_both_observers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'p';d=root/'d';p.mkdir();d.mkdir()
            cfg=root/'controls.json';cfg.write_text(json.dumps({role:dict(root=str(path),folder='audit') for role,path in [('producer',p),('consumer',d)]}))
            with self.assertRaisesRegex(RuntimeError,'request failed'):
                with armed(cfg,'abc123',[1,2],'identity'):raise RuntimeError('request failed')
            self.assertFalse((p/'reference-control.json').exists());self.assertFalse((d/'reference-control.json').exists())


if __name__=='__main__':unittest.main()
