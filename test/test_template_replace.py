import os
from pwtools.common import template_replace
pj = os.path.join


def test_dict():
    # default 'dct' mode
    templ_txt = "%(foo)i %(bar)s"
    rules = {'foo': 1, 'bar': 'lala'}
    ref_txt = "1 lala"
    tgt_txt = template_replace(templ_txt, rules, mode='dct')
    assert tgt_txt == ref_txt, ("ref_txt={} "
                                "tgt_txt={}".format(ref_txt, tgt_txt))

def test_mode_txt_conv_true():
    # 'txt' mode, not default, but actually more often used b/c placeholders
    # are much simpler (no type formatting string, just convert values with
    # str() or pass in string-only values in `rules`).
    templ_txt = "XXXFOO XXXBAR"
    rules = {'XXXFOO': 1, 'XXXBAR': 'lala'}
    tgt_txt = template_replace(templ_txt, rules, mode='txt', conv=True)
    ref_txt = "1 lala"
    assert tgt_txt == ref_txt, ("ref_txt={} "
                                "tgt_txt={}".format(ref_txt, tgt_txt))

def test_mode_txt_conv_true_single():
    # string-only is required, note that conv=False
    templ_txt = "XXXFOO"
    rules = {'XXXFOO': str(1)}
    tgt_txt = template_replace(templ_txt, rules, mode='txt', conv=False)
    ref_txt = "1"
    assert tgt_txt == ref_txt, ("ref_txt={} "
                                "tgt_txt={}".format(ref_txt, tgt_txt))

def test_mode_txt_conv_true_warn_more_rules():
    # warn but pass not found placeholders in `rules`
    templ_txt = "XXXFOO XXXBAR"
    rules = {'XXXFOO': 1, 'XXXBAR': 'lala', 'XXXBAZ': 3}
    tgt_txt = template_replace(templ_txt, rules, mode='txt', conv=True)
    ref_txt = "1 lala"
    assert tgt_txt == ref_txt, ("ref_txt={} "
                                "tgt_txt={}".format(ref_txt, tgt_txt))

def test_mode_txt_conv_true_warn_more_placeholders():
    # warn but do duplicate placeholders
    templ_txt = "XXXFOO XXXBAR XXXBAR"
    rules = {'XXXFOO': 1, 'XXXBAR': 'lala'}
    tgt_txt = template_replace(templ_txt, rules, mode='txt', conv=True)
    ref_txt = "1 lala lala"
    assert tgt_txt == ref_txt, ("ref_txt={} "
                                "tgt_txt={}".format(ref_txt, tgt_txt))

