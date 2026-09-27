# Offline inputs

These small terrain volumes and recorded plans support the offline example, building
checks and regression tests. They are not completed demo worlds or automatic answers
to new requests. Tests and commands load them through the fixture loader; generated
results go under ignored out/ directories.

Keep files that maintained commands consume. Remove obsolete inputs together with the
callers that no longer need them, after checking the package without local caches.
