"""Synthetic cases for isolated Q1 research; never fabricate human study results."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import LABELS, scores, targets
from q1_difficulty_common import align_frame, validate_saved, binary_metrics, subgroup_metrics, bootstrap_f1_gap


class DifficultyDiagnosticsTests(unittest.TestCase):
    def test_fixed_id_alignment_and_duplicate_rejection(self):
        source = pd.DataFrame({'id':['b','a'], 'text':['synthetic B','synthetic A'], 'labels':['[26]','[9]']})
        aligned = align_frame(source, ['a','b'])
        self.assertEqual(aligned.id.tolist(), ['a','b'])
        with self.assertRaises(ValueError):
            align_frame(pd.concat([source,source.iloc[:1]]), ['a','b'])
        with self.assertRaises(ValueError):
            align_frame(source, ['missing'])

    def test_prediction_identity_label_order_and_target_shape(self):
        frame = align_frame(pd.DataFrame({'id':['a','b'], 'text':['synthetic A','synthetic B'], 'labels':['[9]','[26]']}), ['a','b'])
        y = targets(frame)
        saved = dict(sample_id=np.array(['a','b']), labels=np.array(LABELS), true_label_vector=y, probability_vector=y*.8+.1)
        validate_saved(saved, frame)
        for key in ('sample_id','labels','true_label_vector'):
            bad={**saved,key:saved[key][::-1]}
            with self.assertRaises((ValueError,AssertionError)):
                validate_saved(bad, frame)

    def test_counts_match_canonical_f1_and_threshold_boundary(self):
        y=np.array([1,1,0,0]); p=np.array([.5,.1,.9,.2])
        result=binary_metrics(y,p)
        self.assertEqual([result[k] for k in ('tp','fp','fn','tn')],[1,1,1,1])
        self.assertEqual(result['precision'],.5)
        self.assertEqual(result['recall'],.5)
        self.assertEqual(result['f1'],scores(y[:,None],p[:,None])['macro_f1'])

    def test_subgroups_use_positive_and_prediction_denominators(self):
        y=np.array([1,1,0,0,0]);p=np.array([.9,.1,.8,.2,.3])
        r=subgroup_metrics(y,p,np.array([True,False,True,True,True]))
        self.assertEqual((r['n'],r['positive_support'],r['predicted_positive']),(4,1,2))
        self.assertEqual((r['precision'],r['recall'],r['prevalence']),(.5,1.,.25))
        empty=subgroup_metrics(y,p,np.zeros(5,dtype=bool))
        self.assertEqual(empty['n'],0)
        self.assertIsNone(empty['recall'])
        self.assertIsNone(empty['precision'])

    def test_paired_label_gap_bootstrap_is_deterministic(self):
        y=np.array([[1,0],[1,1],[0,1],[0,0]])
        p=np.array([[.9,.1],[.8,.8],[.1,.2],[.2,.9]])
        a=bootstrap_f1_gap(y,p,repeats=200,seed=42)
        self.assertEqual(a,bootstrap_f1_gap(y,p,repeats=200,seed=42))
        self.assertLessEqual(a['ci_low'],a['ci_high'])
        empty=bootstrap_f1_gap(np.zeros((3,2)),np.zeros((3,2)),repeats=200)
        self.assertIsNone(empty['ci_low'])


class AnnotationStudyTests(unittest.TestCase):
    def test_completed_review_schema_and_target_presence_are_validated(self):
        from analyze_q1_annotations import FIELDS, read_reviews
        key=pd.DataFrame({'review_id':['opaque'],'target_label':['gratitude']})
        row=dict(zip(FIELDS,['opaque','gratitude','Human A','yes','explicit','none','','direct_emotion_phrase','complete']))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'review.csv'
            pd.DataFrame([row]).to_csv(path,index=False)
            self.assertEqual(read_reviews(path,key).iloc[0].annotator_id,'human a')
            pd.DataFrame([{**row,'target_presence':'no'}]).to_csv(path,index=False)
            with self.assertRaises(ValueError):read_reviews(path,key)
            pd.DataFrame([{**row,'text':'synthetic private text'}]).to_csv(path,index=False)
            with self.assertRaises(ValueError):read_reviews(path,key)

    def test_disagreement_requires_actual_adjudication(self):
        from analyze_q1_annotations import analyze, blinded_packet
        key=pd.DataFrame([dict(review_id='opaque',sample_id='sample',target_label='gratitude',
                              outcome='TP',truth=1,predicted=1,inclusion_probability=1.)])
        a=blinded_packet(key);a.loc[0,['annotator_id','target_presence','expression','semantic_overlap','status']]=['human-a','yes','explicit','none','complete']
        b=a.copy();b.loc[0,['annotator_id','expression']]=['human-b','implicit']
        adj=blinded_packet(key);adj['resolution_method']=''
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            pending=analyze(key,a,b,adj,out,out,repeats=200)
            self.assertEqual((pending['status'],pending['resolved_n'],pending['disagreement_n']),('pending',0,1))
            adj=a.copy();adj['resolution_method']='third_review'
            with self.assertRaises(ValueError):analyze(key,a,b,adj,out,out,repeats=200)
            adj['annotator_id']='human-c'
            complete=analyze(key,a,b,adj,out,out,repeats=200)
            self.assertEqual((complete['status'],complete['resolved_n']),('complete',1))

    def test_blinded_packet_contains_no_outcome_or_probability_fields(self):
        from analyze_q1_annotations import blinded_packet
        key=pd.DataFrame([dict(review_id='opaque-a',target_label='gratitude',sample_id='source-a',outcome='FP',probability=.99)])
        packet=blinded_packet(key)
        self.assertEqual(packet.review_id.tolist(),['opaque-a'])
        self.assertFalse({'sample_id','outcome','probability','prediction','ground_truth'} & set(packet.columns))
        self.assertTrue(packet.expression.eq('').all())

    def test_recall_excludes_ground_truth_negatives_and_uses_sampling_weights(self):
        from analyze_q1_annotations import expression_recall
        rows=pd.DataFrame([dict(target_label='gratitude',expression='explicit',truth=1,predicted=1,weight=4,outcome='TP'),
                           dict(target_label='gratitude',expression='explicit',truth=1,predicted=0,weight=2,outcome='FN'),
                           dict(target_label='gratitude',expression='explicit',truth=0,predicted=1,weight=100,outcome='FP')])
        result=expression_recall(rows,complete=True,repeats=200)[0]
        self.assertEqual(result['sample_positive_n'],2)
        self.assertEqual(result['sample_recall'],.5)
        self.assertAlmostEqual(result['weighted_recall'],4/6)
        self.assertIsNone(expression_recall(rows,complete=False,repeats=200)[0]['weighted_recall'])

    def test_undefined_kappa_is_not_fabricated(self):
        from analyze_q1_annotations import agreement_metrics
        result=agreement_metrics(['explicit']*3,['explicit']*3)
        self.assertEqual(result['agreement'],1.)
        self.assertIsNone(result['kappa'])


class InterventionTests(unittest.TestCase):
    def test_missing_private_checkpoint_fails_before_inference_imports(self):
        from evaluate_q1_interventions import infer
        with self.assertRaisesRegex(FileNotFoundError,'--checkpoint'):
            infer(None,None,None,None,None,None,None,None)

    def test_matched_controls_use_only_shared_originals_and_remove_stale_output(self):
        from evaluate_q1_interventions import analyze_pairs
        pairs=pd.DataFrame([dict(pair_id='p1',original_sample_id='a',target_label='gratitude',transformation_type='lexical_cue_removal'),
                            dict(pair_id='p2',original_sample_id='a',target_label='gratitude',transformation_type='neutral_control'),
                            dict(pair_id='p3',original_sample_id='b',target_label='gratitude',transformation_type='lexical_cue_removal')])
        original=np.full((3,27),.8);changed=original.copy();changed[:,LABELS.index('gratitude')]=[.3,.7,.2]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);analyze_pairs(pairs,original,changed,out)
            control=pd.read_csv(out/'matched_control_comparison.csv')
            self.assertEqual(control.iloc[0].n_matched_originals,1)
            self.assertAlmostEqual(control.iloc[0].mean_control_adjusted_shift,-.4)
            # A later valid study with no controls must not retain an old control result.
            analyze_pairs(pairs.iloc[[0,2]],original[[0,2]],changed[[0,2]],out)
            self.assertFalse((out/'matched_control_comparison.csv').exists())

    def test_two_human_verified_pair_is_accepted_but_wrong_target_is_rejected(self):
        from evaluate_q1_interventions import PAIR_FIELDS, verified_pairs, text_hash
        row={c:'' for c in PAIR_FIELDS}
        row.update(pair_id='p',original_sample_id='a',paraphrase_id='v',target_label='gratitude',
                   transformation_type='neutral_control',status='verified',verifier_1='human-a',verifier_2='human-b',
                   reviewer_type_1='human',reviewer_type_2='human',semantic_equivalence_1='yes',semantic_equivalence_2='yes',
                   label_preserved_1='yes',label_preserved_2='yes',original_sha256=text_hash('synthetic original'),paraphrase_sha256=text_hash('synthetic variant'))
        truth={'a':np.eye(27,dtype=int)[LABELS.index('gratitude')]}
        self.assertEqual(len(verified_pairs(pd.DataFrame([row]),truth)),1)
        with self.assertRaises(ValueError):verified_pairs(pd.DataFrame([row]),{'a':np.zeros(27)})

    def test_paired_probability_shifts_flips_and_cluster_weighting(self):
        from evaluate_q1_interventions import paired_summary
        r=paired_summary(np.array([.8,.8,.8]),np.array([.2,.7,.9]),['a','a','b'],repeats=200)
        self.assertEqual((r['n_pairs'],r['n_originals'],r['flip_count']),(3,2,1))
        # Original a contributes its mean (-.35), b contributes +.1 equally.
        self.assertAlmostEqual(r['mean_probability_shift'],-.125)
        self.assertAlmostEqual(r['flip_rate'],.25)
        self.assertEqual(r['loss_denominator_pairs'],3)
        self.assertAlmostEqual(r['positive_to_negative_rate_pairs'],1/3)

    def test_unverified_or_ai_only_pairs_cannot_enter_evaluation(self):
        from evaluate_q1_interventions import PAIR_FIELDS, verified_pairs
        row={c:'' for c in PAIR_FIELDS}
        row.update(pair_id='p',original_sample_id='a',paraphrase_id='v',target_label='gratitude',transformation_type='neutral_control',status='proposed')
        truth={'a':np.eye(27,dtype=int)[9]}
        self.assertEqual(len(verified_pairs(pd.DataFrame([row]),truth)),0)
        row.update(status='verified',verifier_1='a',verifier_2='b',reviewer_type_1='ai',reviewer_type_2='human',
                   semantic_equivalence_1='yes',semantic_equivalence_2='yes',label_preserved_1='yes',label_preserved_2='yes')
        with self.assertRaises(ValueError):verified_pairs(pd.DataFrame([row]),truth)

    def test_original_prediction_reproduction_is_mandatory(self):
        from evaluate_q1_interventions import verify_original_scores
        baseline={'a':np.repeat(.2,27)}
        verify_original_scores(['a'],np.repeat(.2,27)[None,:],baseline)
        with self.assertRaises(ValueError):verify_original_scores(['a'],np.repeat(.3,27)[None,:],baseline)

    def test_changed_paraphrase_invalidates_prior_human_verification(self):
        from evaluate_q1_interventions import PAIR_FIELDS, text_hash, validate_text_bindings, bind_texts
        row={c:'' for c in PAIR_FIELDS}
        row.update(pair_id='p',original_sample_id='a',paraphrase_id='v',target_label='gratitude',transformation_type='neutral_control',
                   status='verified',original_sha256=text_hash('synthetic original'),paraphrase_sha256=text_hash('synthetic approved'),
                   verifier_1='reviewer-a',verifier_2='reviewer-b')
        pairs=pd.DataFrame([row]);frame=pd.DataFrame({'id':['a'],'normalized':['synthetic original']})
        payload=pd.DataFrame({'paraphrase_id':['v'],'text':['synthetic changed']})
        with self.assertRaises(ValueError):validate_text_bindings(pairs,payload,frame)
        rebound=bind_texts(pairs,payload,frame)
        self.assertEqual(rebound.iloc[0].status,'proposed')
        self.assertEqual(rebound.iloc[0].verifier_1,'')
