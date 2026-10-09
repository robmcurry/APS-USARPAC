# ui/ — UI build team

Everything the UI team needs to start. Read in order:

1. `handoff/01_brief.md`: what we are building, for whom, and what it is not
2. `handoff/02_user_stories.md`: stories with acceptance criteria
3. `handoff/03_data_contract.md`: the data files the UI reads, and what the solver does not yet export
4. `handoff/samples/run_sample.json`: a real-data mock payload for development

Coming: architecture diagram and a Now/Next/Later roadmap.

The UI never reads solver output directly. A small exporter, to be written in `chapter2_stochastic/`, will produce the bundle described in the data contract. Application code will live in this folder.
