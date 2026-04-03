# Path And Scope Standard

## Series Scoping

- All outputs must be series-scoped
- Never mix assets or outputs across different series
- Infer the series name from `script/<剧名>/...` whenever possible

## Canonical Output Targets

- Director analysis: `outputs/<剧名>/<集数>/01-director-analysis.md`
- Storyboard prompts: `outputs/<剧名>/<集数>/02-seedance-prompts.md`
- Character prompts: `assets/<剧名>-gpt/character-prompts.md`
- Scene prompts: `assets/<剧名>-gpt/scene-prompts.md`
- Explosive rewrite: `script/<剧名>/<集数>__openai__<model>__explosive.md`

## Artifact Priority

Use upstream artifacts in this order:

1. Explosive rewrite if the user wants highest retention/click-through
2. Director analysis for art and storyboard stages
3. Current episode analysis and series context
4. Existing assets of the same series

## Continuity

- Respect previously written series files
- Prefer extending same-series assets instead of regenerating history
- If prior files exist, use them as continuity anchors
