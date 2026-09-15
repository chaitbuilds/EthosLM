"""Public pipeline API; implementation is divided by responsibility."""

import sys as _sys

from types import ModuleType as _ModuleType

_owners = {}

from . import round as _round

from .round import (
    ROOT,
    settlement_site,
    Round,
    OfflineBackend,
    LiveBackend,
    _report_path,
    stage_report,
    _place_stage,
    DETERMINISTIC,
    PLACE,
    default_stages,
    _stopped,
    _needs_model,
    run,
)

_owners.update({n: _round for n in ('ROOT', 'settlement_site', 'Round', 'OfflineBackend', 'LiveBackend', '_report_path', 'stage_report', '_place_stage', 'DETERMINISTIC', 'PLACE', 'default_stages', '_stopped', '_needs_model', 'run')})

from . import stages_plan as _stages_plan

from .stages_plan import (
    PART_GEOMETRY,
    PART_LEAF_KINDS,
    PART_GROUP_KINDS,
    plan_parts,
    plan_plots,
    part_rect,
    part_rects,
    plan_ground,
    PLAN_CHECKS,
    plan_failures,
    _rects_overlap,
    needs_table,
    part_registry_row,
    fixture_part,
    _type_forbidden,
    TYPE_FORBIDDEN,
    TYPE_DECLARATIONS,
    FORMS,
    UNIVERSAL_FORMS,
    read_form,
    form_ok,
    ROLES,
    UNIVERSAL_ROLES,
    COMPOUND_ROLES,
    read_role,
    role_ok,
    NEEDS_DEFAULT,
    GROUND_CLASSES,
    GROUND_OF_SITING,
    read_needs,
    needs_footprint_failure,
    ground_class,
    load_type,
    check_params,
    param_combinations,
    SWEEP_MAX,
)

_owners.update({n: _stages_plan for n in ('PART_GEOMETRY', 'PART_LEAF_KINDS', 'PART_GROUP_KINDS', 'plan_parts', 'plan_plots', 'part_rect', 'part_rects', 'plan_ground', 'PLAN_CHECKS', 'plan_failures', '_rects_overlap', 'needs_table', 'part_registry_row', 'fixture_part', '_type_forbidden', 'TYPE_FORBIDDEN', 'TYPE_DECLARATIONS', 'FORMS', 'UNIVERSAL_FORMS', 'read_form', 'form_ok', 'ROLES', 'UNIVERSAL_ROLES', 'COMPOUND_ROLES', 'read_role', 'role_ok', 'NEEDS_DEFAULT', 'GROUND_CLASSES', 'GROUND_OF_SITING', 'read_needs', 'needs_footprint_failure', 'ground_class', 'load_type', 'check_params', 'param_combinations', 'SWEEP_MAX')})

from . import stages_build as _stages_build

from .stages_build import (
    stage_programs,
    _gate_threshold,
    stage_circulation,
    stage_cache,
    stage_briefs,
    stage_finish,
    _wave_brief,
    _wave_check_runs,
    LINT_MARGIN,
    wave_scope,
    stage_waves,
    _cand,
    _cand_ids,
    _cand_arm,
    _cand_program,
    _cand_dir,
    _cand_plots,
    _TYPE_HEADER,
    instantiated_source,
    voice_palette,
    voice_roof,
    SILHOUETTE_PAIR,
    check_voices,
    instantiate,
    _type_instances,
    _types,
    _type_out,
    _type_file,
    _type_sub,
    _type_rows,
    _instance_program,
    TYPE_CONTRACT,
    TYPE_CONTRACT_KINDS,
    type_contract,
    _plot_block,
    _surface_census,
    stage_type_briefs,
    _sited_block,
    _sited_part_block,
    stage_types,
    _measure_instance,
    _REFUSING_CALLS,
    _watching_refusals,
    _instance_form,
    stage_candidates,
    _type_candidate,
    _revise,
    _revise_subjects,
    _revise_dir,
    REVISE_CARRY,
    _revise_brief,
    _draft_row,
    _record_call,
    _check_recorded,
    stage_revise,
    _arms,
    _arm_list,
    _arm_sub,
    _arm_ids,
    _arm_dir,
    _arm_program,
    stage_arms,
)

_owners.update({n: _stages_build for n in ('stage_programs', '_gate_threshold', 'stage_circulation', 'stage_cache', 'stage_briefs', 'stage_finish', '_wave_brief', '_wave_check_runs', 'LINT_MARGIN', 'wave_scope', 'stage_waves', '_cand', '_cand_ids', '_cand_arm', '_cand_program', '_cand_dir', '_cand_plots', '_TYPE_HEADER', 'instantiated_source', 'voice_palette', 'voice_roof', 'SILHOUETTE_PAIR', 'check_voices', 'instantiate', '_type_instances', '_types', '_type_out', '_type_file', '_type_sub', '_type_rows', '_instance_program', 'TYPE_CONTRACT', 'TYPE_CONTRACT_KINDS', 'type_contract', '_plot_block', '_surface_census', 'stage_type_briefs', '_sited_block', '_sited_part_block', 'stage_types', '_measure_instance', '_REFUSING_CALLS', '_watching_refusals', '_instance_form', 'stage_candidates', '_type_candidate', '_revise', '_revise_subjects', '_revise_dir', 'REVISE_CARRY', '_revise_brief', '_draft_row', '_record_call', '_check_recorded', 'stage_revise', '_arms', '_arm_list', '_arm_sub', '_arm_ids', '_arm_dir', '_arm_program', 'stage_arms')})

from . import stages_measure as _stages_measure

from .stages_measure import (
    stage_lint,
    _doors_from_the_lane,
    stage_measures,
    measure_program,
    build_context,
    diagnose_entry,
    ENTRY_TITLE,
    ENTRY_FIX,
    entry_lines,
    standard_report,
    revision_findings,
    _sd,
    _repair_readout,
    stage_arm_measures,
    region_diff,
    _bar_value,
    _meets,
    MEASURE_ALIASES,
    _m_walk_from_outdoors_pct,
    _stair_clear,
    _town_context,
    _m_e002_from_the_lane,
    _m_own_lint_errors,
    _m_builder_calls,
    _m_program_lines,
    _m_writes_covered_over,
    _instance_rows,
    _m_instances_lint_zero,
    _m_instances_walkable,
    _m_tokens_per_building,
    _parts_record,
    _m_structures,
    _m_instantiation,
    _m_hand_programs,
    _m_enclosure,
    _m_tokens_whole_round,
    _m_whole_place_lint_seconds,
    _m_within_type_variation,
    _m_building_calls_per_type,
    _m_place_read,
    _m_site_chosen,
    MEASURES,
    stage_readout,
)

_owners.update({n: _stages_measure for n in ('stage_lint', '_doors_from_the_lane', 'stage_measures', 'measure_program', 'build_context', 'diagnose_entry', 'ENTRY_TITLE', 'ENTRY_FIX', 'entry_lines', 'standard_report', 'revision_findings', '_sd', '_repair_readout', 'stage_arm_measures', 'region_diff', '_bar_value', '_meets', 'MEASURE_ALIASES', '_m_walk_from_outdoors_pct', '_stair_clear', '_town_context', '_m_e002_from_the_lane', '_m_own_lint_errors', '_m_builder_calls', '_m_program_lines', '_m_writes_covered_over', '_instance_rows', '_m_instances_lint_zero', '_m_instances_walkable', '_m_tokens_per_building', '_parts_record', '_m_structures', '_m_instantiation', '_m_hand_programs', '_m_enclosure', '_m_tokens_whole_round', '_m_whole_place_lint_seconds', '_m_within_type_variation', '_m_building_calls_per_type', '_m_place_read', '_m_site_chosen', 'MEASURES', 'stage_readout')})

from . import stages_media as _stages_media

from .stages_media import (
    stage_cards,
    _resolve,
    stage_judge,
    stage_write,
    stage_render,
    stage_candidate_render,
    stage_type_render,
    _avg_ranks,
    _pair,
    stage_selection,
    _stage_human_look,
    _write_selection,
)

_owners.update({n: _stages_media for n in ('stage_cards', '_resolve', 'stage_judge', 'stage_write', 'stage_render', 'stage_candidate_render', 'stage_type_render', '_avg_ranks', '_pair', 'stage_selection', '_stage_human_look', '_write_selection')})

from . import blind as _blind

from .blind import (
    BUILD_SCRATCH,
    _blind_dir,
    CHECK_LINES,
    _blind_build,
    _collect_blinded,
    scrub_traceback,
    _bounce,
    CHECK_PY,
    CHECK_JOBS,
    _planned_rects,
    stage_check,
    TYPE_CHECK_LINES,
    FIXTURE_ROUNDS,
    FIXTURE_HELD_BACK,
    FIXTURE_RULES,
    plot_ground,
    check_fixtures,
    PART_FIXTURE_ROUND,
    PART_EDGE_CELLS,
    PART_AREA_SIZE,
    PART_EDGE_LOOP,
    PART_EDGE_RELIEF,
    PART_EDGE_CLIFF,
    _free_ground,
    check_parts,
    _fixtures,
    _fixtures_for,
    _seeds_for,
    stage_type_check,
    check_type,
    _fixture_round,
    _run_src,
    _type_findings,
    _type_walkable,
    placed_versus_attempted,
    check_program,
    run_check,
    checked_runs,
)

_owners.update({n: _blind for n in ('BUILD_SCRATCH', '_blind_dir', 'CHECK_LINES', '_blind_build', '_collect_blinded', 'scrub_traceback', '_bounce', 'CHECK_PY', 'CHECK_JOBS', '_planned_rects', 'stage_check', 'TYPE_CHECK_LINES', 'FIXTURE_ROUNDS', 'FIXTURE_HELD_BACK', 'FIXTURE_RULES', 'plot_ground', 'check_fixtures', 'PART_FIXTURE_ROUND', 'PART_EDGE_CELLS', 'PART_AREA_SIZE', 'PART_EDGE_LOOP', 'PART_EDGE_RELIEF', 'PART_EDGE_CLIFF', '_free_ground', 'check_parts', '_fixtures', '_fixtures_for', '_seeds_for', 'stage_type_check', 'check_type', '_fixture_round', '_run_src', '_type_findings', '_type_walkable', 'placed_versus_attempted', 'check_program', 'run_check', 'checked_runs')})

from .stages_plan import (
    stage_place_spec,
    SPEC_BRIEF,
    spec_brief,
    stage_site_search,
    stage_plateau,
    stage_terraces,
    pipeline_site,
    pipeline_voice,
    stage_site,
    stage_plan,
    _record_level,
    _hand_back_level,
    stage_plan_levels,
    plateau_record,
    place_voice,
    _choose_voice,
    _write_registry,
    stage_plan_flat,
    type_declarations,
    _plan_volume,
    validate,
    _hand_back,
)

_owners.update({n: _stages_plan for n in ('stage_place_spec', 'SPEC_BRIEF', 'spec_brief', 'stage_site_search', 'stage_plateau', 'stage_terraces', 'pipeline_site', 'pipeline_voice', 'stage_site', 'stage_plan', '_record_level', '_hand_back_level', 'stage_plan_levels', 'plateau_record', 'place_voice', '_choose_voice', '_write_registry', 'stage_plan_flat', 'type_declarations', '_plan_volume', 'validate', '_hand_back')})

from .stages_build import (
    annotate_gates,
    part_waves,
    instantiate_part,
    stage_parts,
)

_owners.update({n: _stages_build for n in ('annotate_gates', 'part_waves', 'instantiate_part', 'stage_parts')})

from .stages_measure import (
    stage_place_check,
)

_owners.update({n: _stages_measure for n in ('stage_place_check',)})

STAGES = {
    # The top of the plan layer. A sentence becomes a spec, the spec chooses its ground,
    # the ground is levelled where it has to be, and only then is there a site to plan
    # on.
    "place_spec": _place_stage("place_spec"),
    "site_search": _place_stage("site_search"),
    "plateau": _place_stage("plateau"),
    "terraces": _place_stage("terraces"),
    "site": _place_stage("site"),
    "plan": _place_stage("plan"),
    "parts": _place_stage("parts"),
    "place_check": _place_stage("place_check"),
    "circulation": stage_circulation,
    "cache": stage_cache,
    "briefs": stage_briefs,
    "waves": stage_waves,
    "finish": stage_finish,
    "type_briefs": stage_type_briefs,
    "types": stage_types,
    "candidates": stage_candidates,
    "revise": stage_revise,
    "arms": stage_arms,
    "arm_measures": stage_arm_measures,
    "programs": stage_programs,
    "measures": stage_measures,
    "lint": stage_lint,
    "write": stage_write,
    "render": stage_render,
    "candidate_render": stage_candidate_render,
    "type_render": stage_type_render,
    "cards": stage_cards,
    "judge": stage_judge,
    "selection": stage_selection,
    "readout": stage_readout,
}

class _PublicPipeline(_ModuleType):
    """Preserve assignments to the historical public configuration seam."""
    def __setattr__(self, name, value):
        owner = _owners.get(name)
        if owner is not None:
            setattr(owner, name, value)
        super().__setattr__(name, value)

_sys.modules[__name__].__class__ = _PublicPipeline
