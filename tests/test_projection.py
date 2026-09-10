import math
from experiments.project_costs import memory_case,ideal_recall,key_support_probabilities,weighted_quantile


def test_billion_row_payload_rejection_does_not_depend_on_allocator_guesses():
    scan=memory_case(10**9,256,'scan_all',13,13,32e9,1e9)
    branch=memory_case(10**9,256,'branch',13,13,64e9,1e9)
    assert scan['payload_lower_bytes']==40e9 and scan['status']=='payload_exceeds_budget'
    assert branch['payload_lower_bytes']==72e9 and branch['status']=='payload_exceeds_budget'
    assert memory_case(10**9,256,'keys',13,13,64e9,1e9)['status']=='modeled_fit'


def test_support_distribution_counts_query_coordinates_not_document_candidates():
    probabilities=key_support_probabilities(8,2,2)
    assert math.isclose(sum(p for _,p in probabilities),1)
    assert math.isclose(probabilities[0][1],math.comb(6,2)/math.comb(8,2))
    assert weighted_quantile([(10,.6),(1,.4)],.5)==10
    assert weighted_quantile([(10,.4),(1,.6)],.5)==1


def test_ideal_key_recall_is_the_expected_number_of_filled_top_k_slots():
    assert math.isclose(ideal_recall(3,1,2)['expected_recall'],.6875)
