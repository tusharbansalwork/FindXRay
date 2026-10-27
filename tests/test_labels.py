from casefinder.labels import GROUPS, NO_FINDING, OTHER, mesh_to_groups, to_multihot


def test_normal():
    assert mesh_to_groups("normal") == {NO_FINDING}


def test_terms_and_qualifiers():
    g = mesh_to_groups("Cardiomegaly/borderline;Pulmonary Artery/enlarged;Lung/hypoinflation")
    assert g == {"Cardiomegaly", "Low Lung Volume"}


def test_anatomy_without_meaningful_qualifier_is_other():
    assert mesh_to_groups("Lung/hilum") == {OTHER}


def test_ignored_only_is_unusable():
    assert mesh_to_groups("No Indexing") is None
    assert mesh_to_groups("") is None
    assert mesh_to_groups(None) is None


def test_ignored_terms_do_not_block_real_ones():
    assert mesh_to_groups("Technical Quality of Image Unsatisfactory;Pleural Effusion/right") == {
        "Pleural Effusion"
    }


def test_multihot_shape():
    m = to_multihot([{NO_FINDING}, {"Cardiomegaly", "Atelectasis"}])
    assert m.shape == (2, len(GROUPS)) and m.sum() == 3
