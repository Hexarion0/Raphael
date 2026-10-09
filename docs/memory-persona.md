# Memory and personality

RAPHAEL combines a stable companion personality with confirmed personal facts,
recent dialogue, and a running account of older conversations. The upgrade keeps
those sources distinct so continuity does not become invented certainty.

## Personal context you control

These statements start a confirmation before anything is saved permanently:

| Statement | Saved context |
| --- | --- |
| `My current project is Atlas` | Current project |
| `My goal is to finish the voice assistant` | Current goal |
| `I work as a software developer` | Occupation |
| `My favorite game is CS2` | Favorite game |
| `Call me Alex` | Preferred name |

Say `yes` to save or `no` to decline. An unrelated request, cancellation, expiry,
or restart discards an unconfirmed proposal. An explicit `remember ...` command
saves immediately. For example, `remember my current project is Atlas` needs no
second confirmation.

Ask `What's my current project?`, `What's my goal?`, or `What's my occupation?`
for a local answer when a corresponding saved fact exists. After asking about a
supported topic, `No, it's Borealis` proposes a correction to that topic. A full
statement such as `Actually, my current project is Borealis` also works.
Corrections update one current record, with revision history, rather than leaving
competing current values. Use `forget my current project`, `forget my goal`, or
`forget my occupation` to remove the selected record.

Confirmed profile details accompany informal conversation as bounded, dated
context. A saved goal describes what you said when it was recorded; the persona
should check an old plan's relevance before treating it as a present commitment.
Occupation and personal goals describe the user, not system capabilities.
Other notes still use `remember ...`; automatic durable extraction from arbitrary
conversation is not enabled.

## Continuity inside conversations

Short references such as `What about that?`, `Tell me more`, and `Continue` can
use the latest substantive user topic to retrieve relevant saved notes. A clear
new topic uses its own words. Assistant claims are not used to select personal
facts for this contextual lookup. Recall remains local word/alias ranking; it
does not require embeddings, a downloaded memory model, or a separate inference
request. Context is limited to eight recent messages, individual notes are
clipped, and the total recalled-text budget is 6,000 characters.

Older dialogue is summarized into a validated JSON state:

| Field | Purpose |
| --- | --- |
| `overview` | Current topic |
| `user_context` | Attributed goals, constraints, and preferences mentioned in dialogue |
| `decisions` | Choices made by the user |
| `open_threads` | Unanswered questions and unfinished work |
| `superseded` | Corrections and abandoned choices that should not be revived |

The summary instructions ask the provider to remove resolved threads and merge
relevant earlier context. Each field has size and type limits. Invalid JSON,
unexpected fields, empty summaries, and failed requests leave the processing
cursor unchanged for retry. The summary request allows up to 600 output tokens,
up from 150, so there is room for more than a one-sentence recap. Foreground
requests retain priority over this background work.

Existing plain-text summaries remain usable and can be merged into the next
structured summary. Providers returning plain text retain a bounded compatibility
path. Structured state is stored as conversation context, never as automatically
confirmed personal facts. The session identity is unchanged, so the upgrade
continues the existing conversation without deleting or resetting it.

Known forgotten wording is masked in recalled notes as well as conversation
history and summaries. Archived SQLite text remains retained, and paraphrases
outside known wording are not guaranteed to be erased. `/local` restrictions
continue through replies, background summaries, context trimming, and restarts.

## Personality and system prompt

The prompt now composes personality, continuity, speech understanding, capability
limits, runtime context, recalled data, and the user's editable preferences as
separate sections. The default prompt is approximately 1,150 words, compared with
approximately 2,100 previously; custom preferences and recalled data add to it.
This is a reduction in default prompt size, not a measured latency claim.

The intended behavior is warm, mature, feminine, curious, and practical, with
expressive reactions, light wit when welcomed, and reasoned opinions. Replies
should answer the actual request first, build on previous answers, and vary their
wording. Initiative should connect a relevant interest or unfinished thread to
the current conversation, while accepting topic changes and declined questions.
Frustration calls for patient help rather than teasing; brief replies should not
turn every conversation into an interview.

The current request and latest correction take priority over stale context.
Generated summaries are fallible; runtime telemetry and confirmed facts carry
their own sources. Generated replies cannot claim that memories were saved or
actions succeeded without application confirmation. Natural delivery does not
require invented experiences, emotions inferred from raw audio, or shared history.

Your existing `persona.txt` is preserved and continues to load before each reply.
Style preferences there override default delivery examples. The upgrade changes
the application's prompt; it does not replace your private persona file or alter
the voice model. Restart RAPHAEL to load the new application code.

## Conversation acceptance

Automated tests exercise confirmation, corrections, restart recovery, contextual
recall, summary validation, forgetting, and the real CLI callbacks using mocked
providers and audio. They do not prove that a particular provider's dialogue
feels natural. This upgrade passed 768 tests, with three hardware/model integration
tests excluded, plus Ruff and dependency checks. An isolated wheel build and
summary/persona/CLI smoke checks against the installed wheel also passed using
the existing dependency environment. Check these scenarios with your selected model:

1. Save a project and goal, restart, then ask what you're working toward.
2. Ask about a saved note, say `What about that?`, then change to a different topic.
   Expect continuity first and a clean topic change afterward.
3. Make a decision, leave a question unfinished, then continue after a long
   conversation. Expect the decision and open question to remain available.
4. Correct or abandon a plan. Expect the latest wording to guide the answer and
   the old plan to stay abandoned.
5. Discuss an interest, decline a personal follow-up, and give a brief response.
   Expect specific reactions and space, without repeated prompting.
6. Forget a fact that appears in another note, then ask about that note. Expect
   known forgotten wording to remain masked.
