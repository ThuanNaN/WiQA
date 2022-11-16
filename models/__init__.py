# from .QA.viMRC import MRCQuestionAnswering, extract_answer, tokenize_function, tokenize_function_2, \
#                      data_collator, data_collator_2device

# from .QA.viMRC import invoke as viMRC_invoke

# from .QA.albert import invoke as albert_invoke

from .QA.viMRC import viMRC
from .QA.albert import Albert

__all__ = ['viMRC', 'Albert']