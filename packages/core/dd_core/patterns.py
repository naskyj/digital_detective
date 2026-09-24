"""Shared value-shape patterns used by more than one subsystem.

Lives in dd_core precisely because having two copies of the same pattern in
two packages is how a fix lands in one and silently misses the other. That
is not hypothetical: the email pattern existed separately in
`dd_lab/classify/rules.py` (classification) and
`dd_lab/investigate/entities.py` (cross-source correlation). Fixing the
first cleared 24,519 of 24,663 bogus classifications on a real case while
the correlation view stayed full of the identical false positives, because
nothing had touched the second copy.
"""

from __future__ import annotations

import re

#: Extensions that look like a TLD but are a file suffix. `name@2x.png` -
#: the retina-image naming convention - is syntactically identical to
#: `user@domain.tld`, and a naive email pattern matches it. On a real
#: 675,463-artifact case that produced 24,663 artifacts wrongly classified
#: as USER (24,141 of them flagged relevant), and made every one of the top
#: 100 cross-source "correlated identifiers" a bogus email of this shape
#: (alert@2x.png, chat@2x.png, connection@3x.png ...).
_FILE_EXTENSIONS = (
    "png|jpe?g|gif|svg|webp|ico|bmp|tiff?|css|js|mjs|json|xml|html?|"
    "woff2?|ttf|otf|eot|map|mp[34]|wav|ogg|webm|mov|avi|pdf|zip|gz|tar|"
    "exe|dll|dylib|so|bin|dat|log|txt|md"
)

#: An email address whose final label looks like a real TLD rather than a
#: file extension. Deliberately permissive about the local part (the real
#: RFC grammar is far looser than most patterns allow) and strict only
#: about that last label, which is where the false positives came from.
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\."
    rf"(?!(?:{_FILE_EXTENSIONS})\b)"
    r"[A-Za-z]{2,24}\b",
    re.I,
)

#: TLDs common enough to be worth trusting on their own. Deliberately a
#: allow-list rather than an exhaustive IANA list: the point is not to
#: recognise every valid domain, it is to reject strings that merely have
#: an email SHAPE.
_COMMON_TLDS = (
    "com|net|org|edu|gov|mil|int|info|biz|name|pro|xyz|online|site|tech|store|"
    "app|dev|cloud|ai|io|co|me|tv|cc|ly|gg|fm|uk|de|fr|jp|ru|br|au|ca|in|it|nl|"
    "se|no|es|ch|be|at|dk|fi|pl|pt|gr|cz|ie|nz|za|mx|ar|cl|kr|cn|tw|hk|sg|my|"
    "th|vn|ph|id|tr|il|ae|sa|eg|ng|ke|ua|ro|hu|bg|hr|rs|sk|si|lt|lv|ee|is|lu"
)

#: Stricter email pattern for CROSS-SOURCE CORRELATION, which has a very
#: different cost profile from classification. A false positive in
#: classification is one mislabelled row the relevance model can still veto;
#: a false positive in correlation becomes a whole "identifier seen across
#: multiple apps" entity with up to 50 citations, and a handful of them
#: crowd every real finding off the page.
#:
#: Carved memory strings are the problem: raw binary readily produces
#: email-SHAPED fragments (`4@p.amp`, `9@o.ad`, `h@fb.vuw`, `n@2.nc` were
#: all real examples from one case, alongside `edge@2x.pn` - `.pn` being a
#: genuine ccTLD, so no blocklist can reject it). Requiring a plausible TLD,
#: a local part of 2+ chars, and a domain label that is not a bare
#: `<digit>x` retina marker rejects those while keeping real addresses.
#:
#: The trade-off is explicit: an address on an exotic TLD would be missed by
#: correlation (it is still classified as USER by EMAIL_RE above). Extend
#: _COMMON_TLDS if a case needs one.
EMAIL_STRICT_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]{2,}@"
    r"(?!\d+x\b)[A-Za-z0-9-]{2,}(?:\.[A-Za-z0-9-]+)*\."
    rf"(?:{_COMMON_TLDS})\b",
    re.I,
)

#: HTTP entity headers whose names begin with "content". They describe a
#: fetched resource - its MIME type, its size, its encoding - and are never
#: the resource itself, but a bare "content" substring match cannot tell
#: them apart from `message.content`. On a real case `cache.content_type`
#: (values like `image/png`, `application/json; charset=utf-8`) was 5,891
#: artifacts classified CHAT_CONTENT, the single largest false grouping in
#: the category.
_CONTENT_HEADER_SUFFIXES = (
    "type|length|encoding|disposition|range|language|location|md5|"
    "security[_-]?policy"
)

#: "content" used as a real content-bearing field name. Two guards, and
#: both are load-bearing on real data:
#:
#: - not part of a longer word (`AdContentSeenStore`,
#:   `DismissibleContentFrameworkStore` - Discord config stores, not
#:   messages), hence the lookbehind and the `[A-Za-z0-9]` lookahead, which
#:   also rejects camelCase `contentType`;
#: - not an HTTP entity header, hence the suffix lookahead - `_` and `-` are
#:   why `\bcontent\b` alone is not enough: `\b` sees a boundary before the
#:   `-` in `content-type` and (being a word character) no boundary at all
#:   inside `content_type`, so it accepts both.
#:
#: `embedded_content`/`attachment_content` style names are deliberately
#: still matched (an underscore before "content" is a separator, not a word
#: continuation) - they are real content fields.
CONTENT_FIELD_PAT = (
    r"(?<![A-Za-z0-9])content"
    r"(?![A-Za-z0-9])"
    rf"(?![_-](?:{_CONTENT_HEADER_SUFFIXES})\b)"
)
CONTENT_FIELD_RE = re.compile(CONTENT_FIELD_PAT, re.I)

#: The same exclusion for callers matching against a NORMALIZED key tail -
#: lower-cased with every punctuation character stripped (see
#: `dd_lab.classify.rules._norm_key_tail`, and its note on why that
#: substrate deliberately has no word boundaries left to anchor on). A
#: regex with lookarounds is useless there, so the header names are listed
#: in their already-normalized form instead.
HTTP_ENTITY_HEADER_TAILS = (
    "contenttype", "contentlength", "contentencoding", "contentdisposition",
    "contentrange", "contentlanguage", "contentlocation", "contentmd5",
    "contentsecuritypolicy",
)
