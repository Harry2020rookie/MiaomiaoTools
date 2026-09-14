from blackstream.map_model import FloorMapState
from rules.rule_loader import load_rules


def line_state(project_root):
    state = FloorMapState.empty(4, 12, 1)
    for slot in state.slots.values():
        slot.present = True
        slot.node_type = '未知的凶戾'
    for edge in state.edges.values():
        edge.present = True
    state.start_cell = (0, 0)
    state.slots[(0, 0)].node_type = '起点'
    state.slots[(1, 0)].flow_resident = True
    return state, load_rules(project_root / 'data/rules')


def test_unknown_movement_does_not_exclude_distant_base(project_root):
    state, rules = line_state(project_root)
    state.recompute(rules)
    assert state.slots[(10, 0)].settlement_candidate
    assert not state.slots[(3, 0)].settlement_candidate
    assert '“居民”据点' in state.slots[(10, 0)].candidates
    assert state.slots[(10, 0)].inferred_type is None


def test_initial_and_later_screenshot_radius(project_root):
    state, rules = line_state(project_root)
    state.resident_moves = 0
    state.recompute(rules)
    assert state.slots[(6, 0)].settlement_candidate
    assert not state.slots[(7, 0)].settlement_candidate
    state.resident_moves = 2
    state.recompute(rules)
    assert state.slots[(8, 0)].settlement_candidate
    assert not state.slots[(9, 0)].settlement_candidate


def test_unknown_origin_does_not_generate_distance_predictions(project_root):
    state, rules = line_state(project_root)
    state.start_cell = None
    state.recompute(rules)
    assert all(slot.distance is None and not slot.candidates for slot in state.slots.values())
    assert state.prediction_warnings


def test_source_and_visible_base_exclude_candidates(project_root):
    state, rules = line_state(project_root)
    state.slots[(5, 0)].ideal_source = True
    state.recompute(rules)
    assert not state.slots[(5, 0)].settlement_candidate
    state.slots[(6, 0)].node_type = '“居民”据点'
    state.recompute(rules)
    assert not any(slot.settlement_candidate for slot in state.slots.values())


def test_no_residents_does_not_prove_absence(project_root):
    state, rules = line_state(project_root)
    state.slots[(1, 0)].flow_resident = False
    state.resident_moves = 0
    state.recompute(rules)
    assert state.slots[(10, 0)].settlement_candidate


def test_explored_map_does_not_force_missing_minimum(project_root):
    from dataclasses import replace
    from rules.rule_loader import GenerationLimit
    from rules.candidate_filter import propagate_global_capacities
    rules = load_rules(project_root / 'data/rules')
    rules = replace(rules, categories={'unknown_mystery': ['不期而遇', '失与得']},
                    generation_limits={
                        '不期而遇': {'IV': GenerationLimit(1, 1)},
                        '失与得': {'IV': GenerationLimit(0, 1)},
                    })
    predictions = {'remaining': ['不期而遇', '失与得']}
    assert propagate_global_capacities(rules, floor=4, predictions=predictions) == {'remaining': ['不期而遇']}
    assert propagate_global_capacities(rules, floor=4, predictions=predictions,
                                      complete_initial_map=False) == predictions
    warnings = []
    result = propagate_global_capacities(rules, floor=4, predictions={'remaining': ['失与得']}, warnings=warnings)
    assert result == {'remaining': ['失与得']}
    assert warnings


def test_unmatched_screenshot_keeps_actor_separate_from_start(project_root, monkeypatch):
    from blackstream.slot_recognizer import SlotRecognizer
    recognizer = SlotRecognizer(project_root)
    monkeypatch.setattr(recognizer, '_reconcile_reference_topology', lambda state: None)
    state = recognizer.recognize_path(project_root / 'data/screen/macos-auto/IMG_3035.PNG', 3)
    assert state.start_cell is None
    assert all(slot.distance is None for slot in state.slots.values())
    assert state.prediction_warnings


def test_prediction_context_edit_can_be_undone(project_root, monkeypatch):
    from PySide6.QtWidgets import QApplication, QInputDialog
    from blackstream.window import StandaloneWindow
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (2, True))
        window._edit_prediction_context()
        assert window.state.resident_moves == 2
        window._undo()
        assert window.state.resident_moves is None
    finally:
        window.close()
