# test-drive: 90-second demo storyboard

**Hook line (on screen, 0:00):** *1,700 plugins. Don't guess which one helps. Test-drive it.*

Setup before recording: Hermes with test-drive enabled, an API-key model (the recording used
`deepseek/deepseek-v4-flash` through an OpenAI-compatible endpoint), a terminal at 120×36,
and the catalog cache warm (run `/testdrive culture` once). Keep `/testdrive history` empty.

| Time | Screen | Voice-over / caption |
|---|---|---|
| 0:00–0:08 | Title card, then the Hermes CLI | "Hermes has a huge plugin catalog. The hard part is knowing which plugin actually helps." |
| 0:08–0:22 | Type: *"I'm a German manager starting with a Japanese team. Use a cultural-dimensions dataset to rank the Meyer culture-map gaps between Germany and Japan."* Hermes replies that it has **no such tool**. | "Here Hermes hits a wall." |
| 0:22–0:30 | The same reply ends with the 🧪 line: **Global Business Culture**, community, pinned 49218a35, no keys needed, `/testdrive run global-business-culture`. Zoom on it. | "test-drive spots the gap locally, with no extra model call, and names one catalog plugin that fits." |
| 0:30–0:40 | Type `/testdrive culture gap analysis Germany Japan`. Three trust cards appear; highlight *tier · pinned sha · Keys needed: none · Declares: tools culture_gap_analysis…* | "Top 3 matches, each with a trust card taken straight from the catalog entry." |
| 0:40–0:48 | Type `/testdrive run global-business-culture`. Hermes's picker asks: *"download it … into a throwaway Hermes home … then clean up"* → choose **Run the test drive**. | "Nothing downloads until you say yes." |
| 0:48–1:05 | Speed up the spinner (show elapsed ~55 s, sped up 4×). Optional split screen: `ls /tmp/hermes-testdrive-*` shows the sandbox, then it's gone. | "It builds a throwaway Hermes, installs the plugin at its pinned commit, runs your original request once, then attempts to clean up the sandbox." |
| 1:05–1:20 | The report: **Without plugin:** "I have no … tool" · **With global-business-culture:** the ranked table (Evaluating 68, Communicating 65, Disagreeing 58…) · **Evidence: it called culture_gap_analysis** · **Sandbox: deleted ✓** | "A real before/after on *your* request, with proof that the plugin's own tool did the work." |
| 1:20–1:27 | Type `/testdrive keep global-business-culture` → Hermes's own install prompt/card. Then `/testdrive history` shows the receipt. | "Keep it through Hermes's own installer, or discard it. Every drive leaves a receipt." |
| 1:27–1:30 | End card: `test-drive` · github.com/Xuefeng-Zhu/hermes-test-drive · @NousResearch | "test-drive. Try before you install." |

Shot notes
* Use the real run's output. Don't fake the table: run it live and speed up the wait.
* If you have time, add a 3-second cut of the honest case: a plugin whose tools went unused gets
  *"it may not help with this request … /testdrive discard"*.
* Mention once, on screen: "sandbox = throwaway Hermes profile; candidate code runs as your user; cleanup is attempted".
