from utils.pycocoevalcap.bleu.bleu import Bleu
from utils.pycocoevalcap.rouge.rouge import Rouge
from utils.pycocoevalcap.cider.cider import Cider
from utils.pycocoevalcap.meteor.meteor import Meteor


def metric_calculate(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores