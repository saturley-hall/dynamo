import pytest

def test_failing3():
    raise Exception()
    
def test_failing4():
    assert False


def test_failing5():
    assert False
    
def test_failing6():
    pass