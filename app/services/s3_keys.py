"""S3 key arithmetic for filing a document into its section.

Ports of Spring's ``S3StorageService.keyForType`` and ``previewKeyFor`` (see
``D:\\mhn-spring/.../modules/s3/S3StorageService.java``). They are ports, not our own
convention: Spring's user-driven mover writes keys this way, and a document filed by this
service must be indistinguishable from one filed by hand. Diverging here would put two key
conventions in one section table and break any lifecycle rule written against a prefix.

Pure string work, no I/O — the S3 calls live in ``app.integrations.s3``.
"""


def key_for_section(key: str, section: str) -> str:
    """``unclassified/<name>`` -> ``<section>/<name>``.

    Splits on the FIRST slash only, so a nested name survives. A key with no slash is
    treated as a bare object name.
    """
    slash = key.find("/")
    name = key[slash + 1 :] if slash >= 0 else key
    return f"{section}/{name}"


def preview_key_for(key: str) -> str:
    """The sibling preview key: ``reports/<name>`` -> ``reports_preview/<name>``.

    Always derived, never stored — Spring re-derives it at read time, so the two objects
    must keep sharing a name.
    """
    slash = key.find("/")
    if slash < 0:
        return f"{key}_preview"
    return f"{key[:slash]}_preview{key[slash:]}"
