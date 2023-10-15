import abc
from collections import OrderedDict
import numpy as np

"""Defines functions to process the outputs to make them ready for the evaluation."""


def string_to_float(string, default=-1., **unused_kwargs):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


class PostProcessor(abc.ABC):
    """Postprocess the predictions and labels to make them suitable for
    evaluation."""

    def __init__(self, tokenizer, ignore_pad_token_for_loss):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def process(self, preds, labels, data_info=None):
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        return decoded_preds, decoded_labels


class MultiRC(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info)
        preds = [{"group": info["group"], "value":pred} for info, pred in zip(data_info, preds)]
        labels = [{"group": info["group"], "value": label} for info, label in zip(data_info, labels)]
        return preds, labels


class Record(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info)
        labels = [info["answers"] for info in data_info]
        return preds, labels


POSTPROCESSOR_MAPPING = OrderedDict(
    [
        ('superglue-record', Record),
        ('superglue-multirc', MultiRC)
    ]
)


class AutoPostProcessor:
    @classmethod
    def get(self, task, tokenizer, ignore_pad_token_for_loss):
        if task in POSTPROCESSOR_MAPPING:
            return POSTPROCESSOR_MAPPING[task](tokenizer, ignore_pad_token_for_loss)
        return PostProcessor(tokenizer, ignore_pad_token_for_loss)


def wsc_simple(prediction, example_target=None):
  determiners = {
      "a", "an", "few", "her", "his", "each", "every", "many", "much", "my",
      "our", "some", "that", "the", "their", "these", "this", "those", "which",
      "whose", "your"
  }

  def clean(s):
    """Ignore capitalization and determiners."""
    s = s.strip().lower()
    return " ".join([w for w in s.split(" ") if w not in determiners])

  prediction = clean(prediction)
  if not prediction:
    # We don't want an empty prediction to accidentally return 0 and spuriously
    # match the label.
    return -1

  # We aren't using the label but rather using the extracted referent so that we
  # can see if the prediction is equivalent to the referent.
  referent = clean(example_target)

  if ("'" in prediction) != ("'" in referent):
    # Make sure we don't mark cases where the prediction is "Bob" and the
    # referent is "Bob's hat" as predicting the referent.
    predicted_referent = False
  else:
    prediction_words = set(prediction.split(" "))
    referent_words = set(referent.split(" "))

    # Handle cases where the prediction is "fuzzy bunny" and the referent is
    # "bunny".
    predicted_referent = prediction_words.issubset(
        referent_words) or referent_words.issubset(prediction_words)

  return int(predicted_referent)