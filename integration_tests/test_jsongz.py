import pytest
import os
from bigfeta import jsongz
import shutil
import json
import itertools

dname = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.join(
        dname, 'test_files', 'rough_input_tiles.json')
MATCHES = os.path.join(
        dname, 'test_files', 'rough_input_matches.json')


@pytest.mark.parametrize(
        "compress, FILE",
        list(itertools.product([True, False], [TILES, MATCHES])))
def test_jsongz(tmpdir, compress, FILE):
    tmp_file_dir = str(tmpdir.mkdir('file_test_dir'))
    with open(FILE, 'r') as f:
        j = json.load(f)
    tmpfile = os.path.join(tmp_file_dir, "tmp.json")
    tmpfile = jsongz.dump(j, tmpfile, compress=compress)
    newj = jsongz.load(tmpfile)
    assert newj == j
    shutil.rmtree(tmp_file_dir)
