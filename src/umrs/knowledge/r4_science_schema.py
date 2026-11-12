"""Built-in ``r4-science@1`` ontology, written in LinkML (ADR-116 D-4).

Graphs without an ontology use this vocabulary in the extraction prompt. It
is a superset of the 13 types the prose extractor listed before ADR-116
(concept, method, technique, property, process, model, paper, dataset,
software, application, person, material, finding). Its ``r4_bucket``
annotations reproduce the old ``_ENTITY_TYPES`` rule: the types in that set
map to the ``entity`` node_type, the others to ``concept``; ``organization``
is ``entity`` because its alias ``institution`` was in the set.

The schema lives in a Python module so every install carries it, editable or
not, without package-data configuration.

``unclassified`` is an abstract class (bucket ``concept``): extraction never
assigns it from the vocabulary, but warn-mode extraction stores unmatched
types under that name, and projection through this ontology keeps those
nodes.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from .ontology import Ontology

R4_SCIENCE_LINKML = """\
id: https://w3id.org/r4/ontologies/r4-science
name: r4-science
version: "1"
description: >-
  Built-in R4 vocabulary for scientific and technical documents: methods,
  models, data, software, materials, findings and the people and
  organizations behind them.
prefixes:
  linkml: https://w3id.org/linkml/
  r4: https://w3id.org/r4/ontologies/r4-science/
default_prefix: r4
imports:
  - linkml:types

classes:
  ResearchObject:
    abstract: true
    description: Anything a research document talks about.
  Unclassified:
    is_a: ResearchObject
    abstract: true
    description: An extracted term whose type matched no class.
    annotations:
      r4_bucket: concept
  Concept:
    is_a: ResearchObject
    description: An abstract idea, theory or principle.
    aliases: [idea, notion, theory, principle]
    annotations:
      r4_bucket: concept
  Method:
    is_a: ResearchObject
    description: A way of solving a problem, such as an algorithm or analysis approach.
    aliases: [approach, algorithm, methodology]
    annotations:
      r4_bucket: concept
  Technique:
    is_a: ResearchObject
    description: A practical procedure or experimental technique.
    aliases: [procedure, protocol]
    annotations:
      r4_bucket: concept
  Property:
    is_a: ResearchObject
    description: A measurable characteristic of a material, system or model.
    aliases: [characteristic, attribute, quantity]
    annotations:
      r4_bucket: concept
  Process:
    is_a: ResearchObject
    description: A physical, chemical, biological or computational process.
    aliases: [phenomenon, mechanism, reaction]
    annotations:
      r4_bucket: concept
  Model:
    is_a: ResearchObject
    description: A computational, statistical or machine learning model.
    aliases: [ml model, simulation model, neural network, architecture]
    annotations:
      r4_bucket: entity
  Paper:
    is_a: ResearchObject
    description: A publication such as an article, preprint or report.
    aliases: [publication, article, preprint, report]
    annotations:
      r4_bucket: entity
  Dataset:
    is_a: ResearchObject
    description: A collection of data used or produced by research.
    aliases: [data set, corpus, database, benchmark]
    annotations:
      r4_bucket: entity
  Software:
    is_a: ResearchObject
    description: A program, library, package or code base.
    aliases: [tool, library, package, framework, code, notebook]
    annotations:
      r4_bucket: entity
  Application:
    is_a: ResearchObject
    description: A domain or use case where a method or model is applied.
    aliases: [use case, application area, domain]
    annotations:
      r4_bucket: entity
  Person:
    is_a: ResearchObject
    description: A named person, such as an author or researcher.
    aliases: [author, researcher, scientist]
    annotations:
      r4_bucket: entity
  Material:
    is_a: ResearchObject
    description: A substance, compound or physical material.
    aliases: [compound, chemical, substance, molecule]
    annotations:
      r4_bucket: entity
  Finding:
    is_a: ResearchObject
    description: A result, observation or conclusion reported by the work.
    aliases: [result, observation, conclusion]
    annotations:
      r4_bucket: concept
  Instrument:
    is_a: ResearchObject
    description: A device or piece of equipment used to measure or make something.
    aliases: [device, equipment, apparatus, detector]
    annotations:
      r4_bucket: concept
  Organization:
    is_a: ResearchObject
    description: An institution, laboratory, company or consortium.
    aliases: [institution, university, laboratory, company]
    annotations:
      r4_bucket: entity
  Task:
    is_a: ResearchObject
    description: A problem or objective that methods address.
    aliases: [problem, objective, challenge]
    annotations:
      r4_bucket: concept
  Metric:
    is_a: ResearchObject
    description: A measure used to evaluate results, such as accuracy or error.
    aliases: [measure, score, evaluation metric]
    annotations:
      r4_bucket: concept
  Experiment:
    is_a: ResearchObject
    description: A planned study, trial or simulation campaign.
    aliases: [trial, assay, simulation run]
    annotations:
      r4_bucket: concept
  Sample:
    is_a: ResearchObject
    description: A specimen or subject that is measured or treated.
    aliases: [specimen, subject]
    annotations:
      r4_bucket: concept
  Parameter:
    is_a: ResearchObject
    description: A setting or variable of a method, model or experiment.
    aliases: [hyperparameter, setting, variable]
    annotations:
      r4_bucket: concept

slots:
  uses:
    description: The source uses the target, for example a method uses software or data.
    range: ResearchObject
    aliases: [utilizes, employs, applies]
  produces:
    description: The source produces the target, for example an experiment produces a dataset.
    range: ResearchObject
    aliases: [generates, yields, creates, synthesizes]
  part_of:
    description: The source is a part or component of the target.
    range: ResearchObject
    transitive: true
    aliases: [component of, belongs to, contained in, subset of]
  evaluated_on:
    description: The source method or model is evaluated on the target dataset.
    range: Dataset
    aliases: [tested on, benchmarked on, validated on]
  measures:
    description: The source instrument, metric or experiment measures the target property.
    range: Property
    aliases: [quantifies, detects, characterizes]
  improves_on:
    description: The source improves on or outperforms the target.
    range: ResearchObject
    aliases: [outperforms, extends, better than]
  implements:
    description: The source software implements the target method or model.
    domain: Software
    range: ResearchObject
    aliases: [realizes, implementation of]
  depends_on:
    description: The source requires the target.
    range: ResearchObject
    transitive: true
    aliases: [requires, needs, relies on]
  cites:
    description: The source paper cites the target paper.
    domain: Paper
    range: Paper
    aliases: [references, refers to]
  derived_from:
    description: The source is derived from or based on the target.
    range: ResearchObject
    transitive: true
    aliases: [based on, adapted from, built on]
  compares_with:
    description: The source is compared with the target.
    range: ResearchObject
    aliases: [compared to, compared with, versus]
  applied_to:
    description: The source method or model is applied to the target task, material or application.
    range: ResearchObject
    aliases: [used for, applied in, used in]
  authored_by:
    description: The source paper or software is authored by the target person.
    range: Person
    aliases: [written by, created by]
  related_to:
    description: A relation that none of the other relations describes.
    range: ResearchObject
    aliases: [associated with, linked to, correlates with]
"""


@lru_cache(maxsize=1)
def r4_science_ontology() -> Ontology:
    """Parse the built-in schema once; ``source_format`` is ``builtin``."""
    from .linkml_import import linkml_to_ontology

    return replace(linkml_to_ontology(R4_SCIENCE_LINKML), source_format="builtin")


__all__ = ["R4_SCIENCE_LINKML", "r4_science_ontology"]
