# Changelog

## [2.5.0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.4.0...v2.5.0) (2026-05-26)


### Added

* exclude indexers below a minimum graph-node version ([#160](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/160)) ([81353da](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/81353da1c6c5daf940866eb920e8ca3d237ed141))


### Changed

* **k8s:** update cronjob image to sha-81353da ([#163](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/163)) ([94d646a](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/94d646ab1c88d6f7214010d66887a4628bcebe58))

## [2.4.0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.3.3...v2.4.0) (2026-05-21)


### Added

* **dips:** tolerate both legacy and flat /dips/info shapes ([#157](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/157)) ([5a2ede2](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/5a2ede2146fd91a527419f0f7db0eec7b76eed38))


### Changed

* **k8s:** update cronjob image to sha-5a2ede2 ([#159](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/159)) ([121e3f0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/121e3f085e403a5c2b24f8f2d3cb87970a8a8873))

## [2.3.3](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.3.2...v2.3.3) (2026-05-19)


### Changed

* **k8s:** update cronjob image to sha-06472da ([#155](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/155)) ([8cab092](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/8cab0927e5a3d4c54147d5ebc68255cbf47c7109))

## [2.3.2](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.3.1...v2.3.2) (2026-05-19)


### Fixed

* protect MaxMind license key and clarify GeoIP errors ([#153](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/153)) ([06472da](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/06472dab3efba87eb5ac7fb53402823b92972309))

## [2.3.1](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.3.0...v2.3.1) (2026-05-06)


### Changed

* **deps:** bump the actions group with 2 updates ([#148](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/148)) ([b2e4b52](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/b2e4b52a4b3da050d67ad75e9fabf7239eee3aeb))
* **k8s:** update cronjob image to sha-b2e4b52 ([#150](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/150)) ([f24ddb8](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/f24ddb82366aed62eb410e789faca78e88b3005f))


### CI/CD

* publish multi-arch (amd64+arm64) service and release images ([#146](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/146)) ([386c54f](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/386c54fcf084f34fec5dc53d386c0f97fa36271d))

## [2.3.0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.2.1...v2.3.0) (2026-04-21)


### Added

* **iisa:** add GET /scores endpoint for snapshot read-back ([#144](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/144)) ([7c9e85f](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/7c9e85f89731193199ab863d6baa37f60a93b808))


### Changed

* **cronjob:** make score-computation one-shot, drop HTTP+loop ([#141](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/141)) ([59fc33c](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/59fc33cbd5540c00accd090ee525e1c8f5758e73))
* **k8s:** update cronjob image to sha-59fc33c ([#143](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/143)) ([c865c34](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/c865c34223bcbc636e776e899e63a1f6b4c125bf))

## [2.2.1](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.2.0...v2.2.1) (2026-04-21)


### Fixed

* **k8s:** allow score-computation cronjob to push scores to iisa ([#139](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/139)) ([5b8f43b](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/5b8f43bcc5f39f2849da63ba87563f6f7de93ba7))

## [2.2.0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.1.1...v2.2.0) (2026-04-21)


### Added

* **cronjob:** add progress % and ETA to worker heartbeat ([#135](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/135)) ([0625077](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/0625077684f90530c6222042c63c431a2c0aa0f7))


### Changed

* **k8s:** update cronjob image to sha-0625077 ([#138](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/138)) ([053bc03](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/053bc0355b7ca9cd692cfa324dae96fa7fed8741))


### Tests

* **sync-status:** stub _fetch_all_statuses, not asyncio.run ([#136](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/136)) ([af3cbaf](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/af3cbaf827a0f86722ff8a840866c5b86af3b9ca))

## [2.1.1](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.1.0...v2.1.1) (2026-04-20)


### Changed

* **k8s:** update cronjob image to sha-89cd68e ([#133](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/133)) ([bae3be3](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/bae3be37b9e4633a224221dcf5e799c49fb91b5d))

## [2.1.0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.0.2...v2.1.0) (2026-04-20)


### Added

* **cronjob:** print 120s progress heartbeat from redpanda workers ([#128](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/128)) ([fa11e28](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/fa11e283db530e9483a644951d4da67d40eee4ff))


### Performance

* **cronjob:** cut sample-worker row memory and bound merge peak ([#130](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/130)) ([89cd68e](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/89cd68e38fc25d58572efd47f11d17bce87d8c89))

## [2.0.2](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.0.1...v2.0.2) (2026-04-16)


### Fixed

* **k8s:** add imagePullSecrets to iisa Deployment ([#126](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/126)) ([5e1bc7f](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/5e1bc7fb04a0c14c90d6e3119170b34d43d4e0fc))

## [2.0.1](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v2.0.0...v2.0.1) (2026-04-14)


### Changed

* **k8s:** update cronjob image to sha-fa79810 ([#125](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/125)) ([328fb13](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/328fb139566e9d9ecc1df3e10f3d0a0cd9835c11))


### CI/CD

* bump deprecated github actions off node 20 runtime ([#123](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/123)) ([fa79810](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/fa7981053a221742459c6609cbb4b85b1dfc931a))

## [2.0.0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v1.1.1...v2.0.0) (2026-04-14)


### ⚠ BREAKING CHANGES

* **api:** This consolidates /select-one and /select-many into a single /select-indexers endpoint. The Rust client in dipper-iisa will need updating.

### Added

* add comprehensive logging to selection pipeline ([#67](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/67)) ([abc6d0f](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/abc6d0f8ae106b77e02f994496adf1d025bf4ce4))
* add image tag to docker-compose for local builds ([1d90e58](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/1d90e583750f813db8ee0b12f4943c490a285fab))
* add indexer_scores BigQuery table schema ([#13](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/13)) ([ce4ca2c](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/ce4ca2c53af3421ce6cb4ba198055daeb722d90c))
* add justfile with build-image target ([3ea39ad](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/3ea39ad30f569fd22b73bcf945734a1eb1e5b338))
* add score computation CronJob ([#14](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/14)) ([f6083ad](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/f6083add3664ee78530701062a0c9b10d930730f))
* add sync status fetcher service ([#94](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/94)) ([1e799b4](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/1e799b49e3400dece0df4c0ed288130d13de4021))
* add sync status loader and reverse index ([#93](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/93)) ([4622cd7](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/4622cd7611fb6e3db5711eecc90fef743f58b02d))
* **api:** consolidate endpoints, remove acceptance_latency, simplify selection ([#51](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/51)) ([d09d11c](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/d09d11cf93680b338460868b1859f201485f2db5))
* **ci:** add SHA-based image tags with auto-update of k8s yaml ([#42](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/42)) ([c95e2a1](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/c95e2a1531d4e60a31fea92faf95767af0c3fa59))
* **ci:** migrate container registry from GCR to Docker Hub ([#31](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/31)) ([590d119](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/590d119384bb52855c2995f4f386d568e3e62dd8))
* compute real scores without GeoIP, fall back to neutral latency only ([#73](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/73)) ([0816f83](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/0816f8307380b61da7adf3cbdde34543154e6364))
* **cronjob:** add fail-fast validation for GeoIP, source data, and URL cache ([#41](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/41)) ([917bdfd](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/917bdfde56b9312ee08a0e114cd866481bf5e72c))
* **cronjob:** migrate from DB-IP to MaxMind GeoLite2 ([#40](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/40)) ([383af92](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/383af921c72e40a47742d6e7f7e91d581a65f9e1))
* deploy score computation CronJob to GKE ([#19](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/19)) ([2c3c5dd](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/2c3c5dda715135dbe6b6cd58d1e78eac92fd0fcb))
* fetch prices from indexers `/dips/info` endpoint and use prices in IISA scoring ([#61](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/61)) ([fe0d3c2](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/fe0d3c2894d24dd482932a209e5a102f099a3aba))
* **http:** auto-refresh on startup, remove random fallback ([#49](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/49)) ([1999adf](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/1999adf8d2c155f871c1ae97e074dd5d8614f43c))
* improve pipeline logging and fix lat/lon schema ([#21](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/21)) ([101c3b4](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/101c3b4f83d93ac870e63a0ca831c5bdbb0688f3))
* log selection reasoning for each selected indexer ([#103](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/103)) ([3ccf904](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/3ccf904cb9710899adbfd57f746f410dbc9d5802))
* optimistic DIPs fees in stake_to_fees ([#76](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/76)) ([f9204a5](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/f9204a508519e3f4962c51d338b9208391dabf78))
* optional gateway_id filter on Redpanda consumption ([#84](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/84)) ([cbfc1c9](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/cbfc1c91058f6409a420e432d3ddf4f217ee3bab))
* refactor IISA to read pre-computed scores from BigQuery ([#20](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/20)) ([31586fb](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/31586fb7ec540962218b27b55c129a7b3527e713))
* replace BigQuery with Redpanda for score computation ([#59](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/59)) ([30451e2](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/30451e268cf1b038d078b6678035e4c3c3116ce5))
* scoring service with degraded mode ([#71](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/71)) ([5c5fa39](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/5c5fa3969e9485233146eb4bf09993d2347181f0))
* two-pool selection preferring synced indexers ([#99](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/99)) ([89c8b5c](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/89c8b5cb9e412f0be68e151869f9c8014689c1ef))
* wire sync status into API and selection endpoint ([#100](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/100)) ([3af9796](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/3af9796cd6d802fc3706dafbd4e726f78723b2b6))


### Fixed

* API blocklist field alignment and logging improvements ([#30](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/30)) ([81979ba](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/81979ba77b8f0a9c648afb82c058d88fcb0c03c9))
* auto-reload scores in IISA API after cronjob writes them ([c3d407a](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/c3d407a88b077a48627be09ab45a2e940fcb794e))
* **ci:** create PR instead of pushing directly to main ([#43](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/43)) ([6ad3995](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/6ad39951844f64da2b1d8a01493104faaf88e63b))
* **ci:** repair YAML block-scalar parse error in sync-manifest action ([#115](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/115)) ([ee7d509](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/ee7d5096ab567fb4347dfb1ee5eeb625b28674d7))
* correct declined_indexers type, bump coverage, expand lint scope ([#110](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/110)) ([560aa92](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/560aa9225918b3e935f2a3697691b2952ed72c17))
* **cronjob:** add fail-fast permission validation ([#34](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/34)) ([d160f78](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/d160f788247b53680f678bd1bacc7298b256eb3a))
* **cronjob:** add fail-fast validation and fix multiple bugs ([#27](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/27)) ([6937d02](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/6937d0220241b2f7b45c17c1d0421f2f3aa99075))
* **cronjob:** add timeout protection and improve logging ([#26](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/26)) ([c221ba8](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/c221ba8c832033f1ff1ffc163aec8508521d0a63))
* **cronjob:** delay bigframes import until after auth setup ([#33](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/33)) ([112cfd3](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/112cfd3fedc54afc1079623878eee5209a4b6325))
* **cronjob:** ensure dst_lat/dst_lon are float64 for schema stability ([#45](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/45)) ([2fadc44](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/2fadc44911983c8b657676a04ab77cad1946996c))
* **cronjob:** improve permission error detection ([#38](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/38)) ([237d0d3](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/237d0d373bcdeefb297eefc41ea1e51b5f548122))
* **cronjob:** use bigframes.connect() with explicit context ([#32](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/32)) ([60ef8bd](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/60ef8bdc8023ebd265353da0ed2c8a8c0c3b12f2))
* **cronjob:** use explicit service account auth for BigQuery ([b8223b8](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/b8223b84755799bfaae00419b69ad18d99b242cf))
* extend Redpanda consumer window to include today's data ([#82](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/82)) ([ee45bf3](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/ee45bf3cece6148ee0eccf530ab3ac03444dd218))
* Get `indexer_url` from new table `metrics_subgraph_gateway_logs` ([#18](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/18)) ([edeb5db](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/edeb5db33b4e568a24d8b3328de84e1d3251790c))
* **k8s:** rename cronjob pull secret to match org convention ([#117](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/117)) ([24a2e78](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/24a2e787f2c814f77c01fd029f317987277084e4))
* make score computation deterministic and add gateway ID filter ([#106](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/106)) ([99e1a03](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/99e1a0364f770487a808d617019491c0a5f8a2ef))
* move score computation cronjob to 09:00 UTC ([#56](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/56)) ([89b3f6e](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/89b3f6e178b4507ce0a91fe5dc46220dc1bcc51b))
* **normalization:** handle NA values with optimistic scoring ([#48](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/48)) ([9675643](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/9675643e005bcb307987964fb84a7e9bc3f4a743)), closes [#47](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/47)
* remove nodeSelector for GKE Autopilot compatibility ([#23](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/23)) ([9a3aa4f](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/9a3aa4f2257920834d0a61bd222837e65b816f46))
* use network subgraph for indexer discovery ([#69](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/69)) ([745b307](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/745b3078250269b9d4978d7e0c4bf7f77c76578f))


### Changed

* **6e:** cleanup, refactor, and flatten IISA ([#24](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/24)) ([b13f83b](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/b13f83b4eb4bdb4f7b0445648dca2d6bbdc315d9))
* add missing env vars to K8s manifests ([#85](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/85)) ([13f2cec](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/13f2cec6382513f076029c58b8c6b6fd1a547817))
* **cronjob:** replace ipinfo.io with DB-IP for offline GeoIP ([#29](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/29)) ([355e341](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/355e341384588fb283b1863b0b55dc2b968b85a1)), closes [#16](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/16)
* fix pricing normalisation, removal scoring, and dead IQR code ([#87](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/87)) ([0090a4b](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/0090a4bd470fe9f558d43c735692f9471f881301))
* gitignore .claude/ worktree state ([b81a3bb](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/b81a3bbfedc99711e5e1cdcc004a81680a6fd990))
* **k8s:** align cronjob secret refs with graph-infra ([#66](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/66)) ([f283736](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/f2837360d0715f76fb0683eec91b2811e28fcfc7))
* **k8s:** update cronjob image to sha-2fadc44 ([#46](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/46)) ([5ed4e72](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/5ed4e72408293bf76e1f0be7f22b99ad9e68551e))
* **k8s:** update cronjob image to sha-6ad3995 ([#44](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/44)) ([a61558e](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/a61558eb6fa931480b6d574b27ccb8bcdb1b9b32))
* **k8s:** update cronjob image to sha-d09d11c ([#54](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/54)) ([c61e201](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/c61e20134dba15ba200f02cd1312b54ee8d2a85b))
* push scores to iisa over HTTP, drop shared Filestore PVC ([#119](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/119)) ([57a965b](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/57a965b4e669ce5057f5a0faa4310fab81dfb221))
* remove 3 skipped flaky tests ([#101](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/101)) ([a2f805c](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/a2f805c8f24bd79692cf34e93809005955b38a50))
* remove dead avg_sync_duration column ([#89](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/89)) ([679a302](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/679a3025380507f29ec2000b19040904dc7c0fe6))
* remove dead test code for existing_dips_agreements and orphaned methods ([#80](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/80)) ([2cfc036](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/2cfc0362b847eddc1c7244bde3024a4e7d491e8c))
* remove IQR deviation from stake_to_fees scoring ([#78](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/78)) ([fa4ad70](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/fa4ad70891d071a1b6d4dae7e0ae463efe7b240b))
* rename _notify_iisa_refresh to _refresh_iisa_scores ([e17881b](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/e17881bc0a1fd20473d9d236012e597800057772))
* use GRT per billion entities instead of per million ([#64](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/64)) ([d13fbe8](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/d13fbe856fcf55c33764746082c7b7b2f2b8e919))


### Performance

* use ProcessPoolExecutor for parallel partition consumption ([#97](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/97)) ([818a470](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/818a470644908aa2b68b058d6a38e05858982679))


### Tests

* add coverage for pricing extraction and filtering functions ([#104](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/104)) ([f0e9284](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/f0e9284063881f99d94d087df273d679302ab671))
* **iisa:** add comprehensive tests for iisa_http_endpoints.py ([#35](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/35)) ([4c39404](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/4c39404acb18e51ab11d25e20a4a44b21e110e7a)), closes [#25](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/25)


### CI/CD

* add GHCR workflow for IISA service image ([#63](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/63)) ([bdfac35](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/bdfac35ca9c57fb466ba7ba9c0da66eec2a7dbbd))
* add mypy type checking to CI workflow ([#107](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/107)) ([7ad8e2a](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/7ad8e2a0cae1f4b70123f2e6acff34c186114f93))
* bridge release-please releases to build-release-image via dispatch ([#122](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/122)) ([d652289](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/d652289b72372fe5f02c7501385d180b9d4037d6))
* publish versioned iisa image on v* tag push ([#121](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/121)) ([0b034c1](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/0b034c1e611aaf0f7ee2fef5d60b04f02460c41a))
* rename docker images to match repo name and move cronjob to GHCR ([#113](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/113)) ([3e530ce](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/3e530ceb402ff9d79eee8fd26eef04b027db697a))
* use github.repository for image name ([f40e347](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/f40e347b1f43d0312d3581797cbc62b8f06d1c9b))

## [1.1.1](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v1.1.0...v1.1.1) (2026-01-14)


### Fixed

* **ci:** prevent sync-manifest from downgrading manifest version ([ac8954c](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/ac8954c5b3b17147996819bceca301cb22261031))


### Changed

* auto-sync manifest to 1.0.0 (was 1.1.0) [skip ci] ([2cd2c2d](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/2cd2c2da0e66c801b835c4eb6592959ff8232b78))
* auto-sync manifest to 1.1.0 (was 1.0.0) [skip ci] ([7a7b182](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/7a7b1828fef3e41d82ac6f4c0e6d24d1a76a895b))

## [1.1.0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/compare/v1.0.0...v1.1.0) (2026-01-14)


### Added

* **api:** add blocklist and declined_indexers to selection request ([#8](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/8)) ([0092520](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/0092520f45dd745477b39f6c2a20fb8c5e7c7688))


### Fixed

* **dev:** add missing test dependencies ([#7](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/7)) ([885d70b](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/885d70bfce216de3f0f01b0a2927cd5e94893a11))


### Changed

* auto-sync manifest to 0.0.0 (was 1.0.0) [skip ci] ([ce65387](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/ce65387bbe94338b5bf3f05d67ac8894f443c8a6))
* auto-sync manifest to 1.0.0 (was 0.0.0) [skip ci] ([ed63176](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/ed6317609cf61f586f61bacd8f1ba9aa9292b572))

## 1.0.0 (2026-01-13)


### Added

* **iisa-service:** add FastAPI HTTP service ([bdd4d4e](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/bdd4d4e08eefe144e54110d4c923f890f8f0236f))
* **iisa:** add core IISA library ([1c426d0](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/1c426d06b39ee1dbef30b88659ef5f2b8be4ff56))


### Changed

* auto-sync manifest to 0.0.0 (was 0.1.0) [skip ci] ([9f97100](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/9f9710050b0b42b436ee2d535e2c9c061fe977e6))
* initial project setup and containerization ([f6bffb4](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/f6bffb484acc1ca3323a08fdc5f78fec56821ac4))


### Infrastructure

* **k8s:** add Kubernetes deployment manifests for IISA ([#2](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/2)) ([51ef538](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/51ef53897f9a2ff6a3cc6cac8b2266b3d3a9fb94))


### Tests

* add comprehensive test suite ([b76e390](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/b76e3907a1e19ce95c4c188a5d8e3146a544b341))


### CI/CD

* **release:** add release-please for automated releases ([#3](https://github.com/edgeandnode/subgraph-dips-indexer-selection/issues/3)) ([a4d015c](https://github.com/edgeandnode/subgraph-dips-indexer-selection/commit/a4d015c57620e3107de0cad0ba29da23cc533714))
