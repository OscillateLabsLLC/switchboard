# Changelog

## [1.1.1](https://github.com/OscillateLabsLLC/switchboard/compare/v1.1.0...v1.1.1) (2026-10-15)


### Bug Fixes

* a skill with huge dialog files can no longer crash classifier training ([82d39dc](https://github.com/OscillateLabsLLC/switchboard/commit/82d39dcd31bd466e122b94976fb9f15c576512fd))
* a skill with huge dialog files can no longer crash classifier training ([c7af3f7](https://github.com/OscillateLabsLLC/switchboard/commit/c7af3f7dfa81b4b4d253e2b0a81eeaf9bc135e92))

## [1.1.0](https://github.com/OscillateLabsLLC/switchboard/compare/v1.0.0...v1.1.0) (2026-10-15)


### Features

* collision_resolution "specific" (default) for template collisions ([97906b2](https://github.com/OscillateLabsLLC/switchboard/commit/97906b224ba9a4b3e13d4339882aa94227dc0ec3))
* make collision_resolution "specific" the default ([59bc68a](https://github.com/OscillateLabsLLC/switchboard/commit/59bc68af85c50d79d2814d57d66b38e668e65091))
* optional collision_resolution "specific" for template collisions ([62cde38](https://github.com/OscillateLabsLLC/switchboard/commit/62cde38592c4102c1fbdd28d5aefad9f4fb51483))

## 1.0.0 (2026-10-15)


### Features

* answer OVOS readiness; first build skips the debounce ([ce44dbc](https://github.com/OscillateLabsLLC/switchboard/commit/ce44dbcd46fd7939a5fe6ab2d0f96f6fe5f3e7a2))
* answer OVOS readiness; first build skips the debounce ([49b28e1](https://github.com/OscillateLabsLLC/switchboard/commit/49b28e10375e874a24c0a1bdfe5c0fef8f3b4711))
* catch-all templates yield to a confident classifier ([c7ec5ed](https://github.com/OscillateLabsLLC/switchboard/commit/c7ec5ed83cfeb67ad621dfa9adaa98db689a34e4))
* catch-all templates yield to a confident classifier ([9bf1aca](https://github.com/OscillateLabsLLC/switchboard/commit/9bf1aca1eacc875fb5b2c280f305cfe242666653))
* **exact:** collision_priority, prefer core or user skills on template collisions ([bb6f9b2](https://github.com/OscillateLabsLLC/switchboard/commit/bb6f9b2c4d3986b896ce6473f610a2dde14d9cc6))
* **exact:** collision_priority, prefer core or user skills on template collisions ([290bb34](https://github.com/OscillateLabsLLC/switchboard/commit/290bb344a541372c187b6e8527fd34e0a742174f))
* **exact:** skill_priority, an ordered list of skills that win collisions ([0edcb05](https://github.com/OscillateLabsLLC/switchboard/commit/0edcb05ceb40742defaa85b71fc6c3b9c5c8add2))
* experimental decider step, a local decision model in place of the LLM ([168890e](https://github.com/OscillateLabsLLC/switchboard/commit/168890ea1000dbd31b0b64a31adf0406de949faf))
* experimental decider step, a local decision model in place of the LLM ([c8bb9fe](https://github.com/OscillateLabsLLC/switchboard/commit/c8bb9fe354a4b61ffc32c51a7801f0ab08ef3fbc))
* float32 classifier features (the only model) ([619861c](https://github.com/OscillateLabsLLC/switchboard/commit/619861c1480ec5ab59077ca71484d507f83966ba))
* float32 classifier features (the only model) ([930574d](https://github.com/OscillateLabsLLC/switchboard/commit/930574ded49cf2964b34e9d59a8a7c9056b37984))
* high_risk "confirm" hands held requests to Doublecheck ([a45857a](https://github.com/OscillateLabsLLC/switchboard/commit/a45857a6c64cf362993245408353c5bb63b39215))
* high_risk "confirm" hands held requests to Doublecheck ([d35ef53](https://github.com/OscillateLabsLLC/switchboard/commit/d35ef534aa59ea4afa1ec53d81136cc38b188761))
* OVOS pipeline plugin with Hangup declines and shadow mode ([6b4e1f0](https://github.com/OscillateLabsLLC/switchboard/commit/6b4e1f086768d65d6ab73b5698d12de1dee2986f))
* router library lifted from the intent-eval harness ([81a1a14](https://github.com/OscillateLabsLLC/switchboard/commit/81a1a14d34654b79c50570a64f77a96b4155b70c))
* strip leading wake phrases; optional prefix step for trailing speech ([e7c6e97](https://github.com/OscillateLabsLLC/switchboard/commit/e7c6e97e6e9134aa688836c50df696294b5551fa))
* strip leading wake phrases; optional prefix step for trailing speech ([ad7cdef](https://github.com/OscillateLabsLLC/switchboard/commit/ad7cdefdbed194edfa3248dfa16b41cd5ebef62a))
* Switchboard as four OVOS pipeline steps (echo, exact, classifier, llm) ([d0dbd0c](https://github.com/OscillateLabsLLC/switchboard/commit/d0dbd0c300aca339378731e764709afa28a061c5))
* train the classifier in a child process ([feba61f](https://github.com/OscillateLabsLLC/switchboard/commit/feba61faaca3f5ed33058a5977099b4253259fd8))
* train the classifier in a child process ([f0c9e5d](https://github.com/OscillateLabsLLC/switchboard/commit/f0c9e5db6c5863534119f0fe42735b3265b7e898))


### Bug Fixes

* a held high-risk intent is declined, not passed on ([07fdce4](https://github.com/OscillateLabsLLC/switchboard/commit/07fdce4d1d4d2ad69b815653420b2810b2acc706))
* a held high-risk intent is declined, not passed on ([adfc22e](https://github.com/OscillateLabsLLC/switchboard/commit/adfc22eea282fbd7c54b39117ec9383e5315bcbb))
* **confirm:** hold through Hangup when Doublecheck isn't loaded ([c91929d](https://github.com/OscillateLabsLLC/switchboard/commit/c91929d550c3e915cede9c0a6412128040197e04))
* **decider:** own inference code instead of a lint-excluded vendored file ([d96feb2](https://github.com/OscillateLabsLLC/switchboard/commit/d96feb2824f6c3b9646bef2898319a3d215b1a43))
* decline "not for me" speech the LLM finds no tool for ([88692d6](https://github.com/OscillateLabsLLC/switchboard/commit/88692d6020a6b19a1cb74cce0e6c784d7b991d78))
* decline "not for me" speech the LLM finds no tool for ([eeb315a](https://github.com/OscillateLabsLLC/switchboard/commit/eeb315a10551bcf0591c0af8276c6f165f71a685))
* **echo:** Doublecheck's quote of the user is not echo ([51c3bbd](https://github.com/OscillateLabsLLC/switchboard/commit/51c3bbdb43f68d1bde8c7c9422dec206d6cff25f))
* hand OCP's own intents to OCP's pipeline step ([21e3e00](https://github.com/OscillateLabsLLC/switchboard/commit/21e3e0022bd0d31fd72fb13680910bcc15825d85))
* hand OCP's own intents to OCP's pipeline step ([952acb8](https://github.com/OscillateLabsLLC/switchboard/commit/952acb8690a440d775bce012235cfc0ead325e45))
* keep typed-slot templates when skipping malformed ones ([173235f](https://github.com/OscillateLabsLLC/switchboard/commit/173235fe5180038600210d390dc54e57465d8989))
* live startup on ovos-core; cache the trained classifier ([91d9449](https://github.com/OscillateLabsLLC/switchboard/commit/91d94493234f931709dd2d8bc681279817e45223))
* live startup on ovos-core; cache the trained classifier ([d70466b](https://github.com/OscillateLabsLLC/switchboard/commit/d70466b53db60cbc429e4dfda7d8fea465e3671d))
* **llm:** decline when the model answers with a tool it wasn't offered ([8771268](https://github.com/OscillateLabsLLC/switchboard/commit/87712682240a506d50ebe42a5899aaf51210639e))
* **llm:** decline when the model answers with a tool it wasn't offered ([fb416a1](https://github.com/OscillateLabsLLC/switchboard/commit/fb416a10aef9036395fa8c626b4ce5a8b85d6cde))
* **llm:** give unknown_tool declines the full details the router reads ([9c87c5d](https://github.com/OscillateLabsLLC/switchboard/commit/9c87c5d7e3d5c1cc2dd8ea679691fa3a21bc97ce))
* say "unmapped intent" when the classifier passes an unmapped top class ([f4f412c](https://github.com/OscillateLabsLLC/switchboard/commit/f4f412ceb9068e84c1d9e5c62ec2b2886f7ef139))
* say "unmapped intent" when the classifier passes an unmapped top class ([1bd1023](https://github.com/OscillateLabsLLC/switchboard/commit/1bd10235578870ddfbfce8fa3cf2b62baf429f9b))
* skip malformed templates instead of failing the whole build ([de85cf7](https://github.com/OscillateLabsLLC/switchboard/commit/de85cf7677b64c8a4fbb4038f6c318e59df309ca))
* skip malformed templates instead of failing the whole build ([0a79ceb](https://github.com/OscillateLabsLLC/switchboard/commit/0a79ceb3532e6f77c173b6df1bc7ce0ccee39a6e))
* template words end at word boundaries ([a249387](https://github.com/OscillateLabsLLC/switchboard/commit/a249387ed9a09ef1f22149d13f545f15d7f0c8a3))
* template words end at word boundaries ([b239240](https://github.com/OscillateLabsLLC/switchboard/commit/b23924044a56590fc06b702122cd09c2eef16344))


### Performance Improvements

* **decider:** one packed row per question, 128 -&gt; 80 ms per decision ([77d8764](https://github.com/OscillateLabsLLC/switchboard/commit/77d8764789aa5cdde3d4d4f677303cca5bd161a0))


### Documentation

* confirm-flow design proposal (for review; nothing built) ([f0585a5](https://github.com/OscillateLabsLLC/switchboard/commit/f0585a530d7ffd68af0b68f4477d6d81f7b78315))
* confirm-flow design reflects what was built and decided ([c0c1d5f](https://github.com/OscillateLabsLLC/switchboard/commit/c0c1d5fbe55362b0d8dfcd5dcb57561029349653))
* describe the evaluation without private household details ([6249960](https://github.com/OscillateLabsLLC/switchboard/commit/6249960f84d54fe35c3da9714cdd0f6e37bb2294))
* design for confirming high-risk actions ([535e5ed](https://github.com/OscillateLabsLLC/switchboard/commit/535e5ed847477faa1d94b9099546d5bcb4ae558e))
* Doublecheck's question wording ([658f5e7](https://github.com/OscillateLabsLLC/switchboard/commit/658f5e7e9750a941433b31ca193d351dd5823982))
* dynamic schema proposal for routing arbitrary skill intents ([74eb381](https://github.com/OscillateLabsLLC/switchboard/commit/74eb38125ee3262742e67ec7d6ed7e115ed1b5a3))
* dynamic schema proposal for routing arbitrary skill intents ([527230d](https://github.com/OscillateLabsLLC/switchboard/commit/527230d7be96459979cd4ac299f05610300e2627))
* evaluation results, limits and configurations per situation ([53f7f74](https://github.com/OscillateLabsLLC/switchboard/commit/53f7f74a88c4e64fac34350e7ce6d183cd1a2ec3))
* evaluation results, limits and configurations per situation ([6bee852](https://github.com/OscillateLabsLLC/switchboard/commit/6bee852111146441e20bc1e62d7ee1895a342574))
* **evaluation:** decider latency is ~130 ms with the packaged implementation ([3da9d2a](https://github.com/OscillateLabsLLC/switchboard/commit/3da9d2aded5f6d41729bf9308f245bd014e8404f))
* **evaluation:** desktop numbers re-scored; decider and 1.7B on float32 ([8b52f4c](https://github.com/OscillateLabsLLC/switchboard/commit/8b52f4c35ea3c2713ebacf1ca1eaefd37b55d0f4))
* **evaluation:** desktop numbers re-scored; decider and 1.7B on float32 ([840fa16](https://github.com/OscillateLabsLLC/switchboard/commit/840fa1696cda0de06bba792e9add72b79fc530e9))
* **evaluation:** how it fails (doing nothing, not the wrong thing); decider config; recommendation ([5f64d57](https://github.com/OscillateLabsLLC/switchboard/commit/5f64d570a3e3c05a9b27edc323b6aeede1376b93))
* **evaluation:** measured overheard-speech rates; LLM on CPU not viable; MLX caveat ([32d1144](https://github.com/OscillateLabsLLC/switchboard/commit/32d11441dd6a083f0fdea1b539ed58fa135d146e))
* fix list nesting in the confirm-flow design ([ac5a8d5](https://github.com/OscillateLabsLLC/switchboard/commit/ac5a8d5da89c3de948ea25c9225298790a4c9046))
* known limitations; confirm is opt-in by design ([f0e9e6f](https://github.com/OscillateLabsLLC/switchboard/commit/f0e9e6f5b56e90eb0fea12b0c58eda27fdbd884b))
* known limitations; confirming high-risk actions is opt-in by design ([4c1ee13](https://github.com/OscillateLabsLLC/switchboard/commit/4c1ee13f305c4189c62ef2344c87432b652a2570))
* mini PC latency (p95 9.2-9.5 ms on a 4-core mini PC) ([c34aa6a](https://github.com/OscillateLabsLLC/switchboard/commit/c34aa6ad6f1f668572712b9c642fa5b6907d5dca))
* outperforms Adapt, Padatious and Model2Vec, rather than beats stock OVOS ([7e9ebfb](https://github.com/OscillateLabsLLC/switchboard/commit/7e9ebfb29e1a76d73ebfc39ac602dad364a301e6))
* README, CI workflow and uv.lock ([8c418c4](https://github.com/OscillateLabsLLC/switchboard/commit/8c418c47e15bddd26269f204f205d9abf18663af))
* **readme:** generic LLM endpoint example ([1818193](https://github.com/OscillateLabsLLC/switchboard/commit/18181938fdb6cca2fe113dfe409781e0e54ae4b3))
* **readme:** optional install step for Doublecheck ([bc0a625](https://github.com/OscillateLabsLLC/switchboard/commit/bc0a625893b20c28a4ed25c4abf7e3874fb4c785))
* record overlay and upstream decisions; restate the grouping question ([f74b1ad](https://github.com/OscillateLabsLLC/switchboard/commit/f74b1ad98cc99a397d4f898f3f0ce51629c7ce9c))
* record the agreed answers in the confirm-flow design ([f1acbcf](https://github.com/OscillateLabsLLC/switchboard/commit/f1acbcfa1497c63d1583898451a0da8d28c73624))
* record the grouping rule and its measurement ([450f9cc](https://github.com/OscillateLabsLLC/switchboard/commit/450f9cca30e831fefd33e2eafc95a737c3c2039f))
* restart the listener after changing the pipeline ([d6c4af8](https://github.com/OscillateLabsLLC/switchboard/commit/d6c4af8a6276c9ee2e829b79bbc52e12d4bd80d6))
* rewrite for people new to Switchboard ([12af3dc](https://github.com/OscillateLabsLLC/switchboard/commit/12af3dc67ea7e679ac4a5c87add80a9d1821954c))
* rewrite for people new to Switchboard ([7be4d41](https://github.com/OscillateLabsLLC/switchboard/commit/7be4d41b6e1f7ba719cd0a894734d9df261f8f20))
* tell disambiguation apart from Doublecheck's confirmation ([de2cbe4](https://github.com/OscillateLabsLLC/switchboard/commit/de2cbe4901186c67b90ea37b529dbf5b936e82cd))
* the handler step is LabelSelector.select ([fba137f](https://github.com/OscillateLabsLLC/switchboard/commit/fba137f61340e0a333fccbe112a2e5168dafd791))
* where the design came from ([476b2fb](https://github.com/OscillateLabsLLC/switchboard/commit/476b2fb35ed9922629231303944bdcdea4f359ba))


### Miscellaneous Chores

* release 1.0.0 ([2c1bb5e](https://github.com/OscillateLabsLLC/switchboard/commit/2c1bb5eb5625f66998ef15d83f4dce1ade689c86))
