# Visualization upgrade

This patch adds the governed Matplotlib visualization layer before the repository's first commit.

It introduces:

- five strategic, publication-quality brief figures;
- light-background institutional style controls;
- 300-DPI and pixel-dimension publication gates;
- source, sample, figure ID, and generation-time footers;
- no-placeholder and no-synthetic-chart controls;
- ReportLab integration so each PDF page receives one dominant figure;
- tests for the visualization contract and empty-data rejection.

The patch does not generate charts or claim results. Rendering remains blocked until real pipeline outputs exist.
