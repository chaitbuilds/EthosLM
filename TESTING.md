# Test coverage

The shipped suites cover maintained components. Some cases require a server
or recorded inputs and skip when these are absent. A zero exit status does
not mean every case ran, or that arbitrary requests produce good places.

The development repository retains additional suites excluded from this
package. Failures in that list remain unresolved; they are not passing tests.

| Excluded suite | Reason |
| --- | --- |
| test_architecture | imports test_rings, which does not ship |
| test_place_spec | fails in this repository as well; kept in the record until it is repaired |
| test_closure_meaning | reads a development run's state or frozen agent answers, which do not ship |
| test_compile | fails in this repository as well; kept in the record until it is repaired |
| test_composition_finish | reads a development run's state or frozen agent answers, which do not ship |
| test_composition_section | fails in this repository as well; kept in the record until it is repaired |
| test_rings | fails in this repository as well; kept in the record until it is repaired |
| test_compound_ground | fails in this repository as well; kept in the record until it is repaired |
| test_delivery_form | fails in this repository as well; kept in the record until it is repaired |
| test_design_evidence | fails in this repository as well; kept in the record until it is repaired |
| test_form_plans | reads a development run's state or frozen agent answers, which do not ship |
| test_ground_contract | imports test_rings, which does not ship |
| test_site_needs | fails in this repository as well; kept in the record until it is repaired |
| test_growth | imports test_compile, which does not ship |
| test_loop | fails in this repository as well; kept in the record until it is repaired |
| test_compounds | fails in this repository as well; kept in the record until it is repaired |
| test_neighbourhood | reads a development run's state or frozen agent answers, which do not ship |
| test_neighbourhood_evidence | reads a development run's state or frozen agent answers, which do not ship |
| test_density | fails in this repository as well; kept in the record until it is repaired |
| test_place | fails in this repository as well; kept in the record until it is repaired |
| test_promotion_functions | reads a development run's state or frozen agent answers, which do not ship |
| test_realization | imports test_types, which does not ship |
| test_siting | fails in this repository as well; kept in the record until it is repaired |
| test_solver | fails in this repository as well; kept in the record until it is repaired |
| test_stages | fails in this repository as well; kept in the record until it is repaired |
| test_types | fails in this repository as well; kept in the record until it is repaired |
