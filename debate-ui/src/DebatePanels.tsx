import React from "react";

// -----------------
// Header + Tabs
// -----------------
export function HeaderBar({ baseUrl, setBaseUrl, threadId, setThreadId, loading, onLoad }: any) {
  return (
    <header className="flex flex-col sm:flex-row gap-3 sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-bold">Debate Control Panel</h1>
        <p className="text-sm text-gray-600">Tabbed left pane · Persistent Run & Transcript</p>
      </div>
      <div className="flex flex-col sm:flex-row gap-2">
        <input
          className="border rounded px-3 py-2 w-full sm:w-72"
          placeholder="FastAPI base (…/debate)"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
        />
        <input
          className="border rounded px-3 py-2 w-full sm:w-72"
          placeholder="Thread ID"
          value={threadId}
          onChange={(e) => setThreadId(e.target.value)}
        />
        <button className="px-4 py-2 rounded bg-gray-900 text-white hover:opacity-90" onClick={onLoad} disabled={!threadId || loading}>
          {loading ? "Loading…" : "Load"}
        </button>
      </div>
    </header>
  );
}

export function TabBar({ items, active, onChange }: any) {
  return (
    <div className="mb-3 overflow-x-auto">
      <div className="inline-flex items-center gap-1 border rounded-xl p-1 bg-gray-100">
        {items.map((t: any) => (
          <button
            key={t.key}
            className={`px-3 py-1.5 rounded-lg text-sm whitespace-nowrap ${active === t.key ? "bg-white border shadow-sm" : "hover:bg-white/60"}`}
            onClick={() => onChange(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// -----------------
// Left tabs
// -----------------
export function CreateTypesTab(props: any) {
  const {
    profileTypes,
    loadProfileTypesOrSeed,
    loadProfileTypes,
    typesLoading,
    projectIdNew,
    setProjectIdNew,
    titleNew,
    setTitleNew,
    phaseNew,
    setPhaseNew,
    initialMessageNew,
    setInitialMessageNew,
    contextNew,
    setContextNew,
    selectedTypeKeys,
    setSelectedTypeKeys,
    quickProfileName,
    setQuickProfileName,
    quickProfileTypeKey,
    setQuickProfileTypeKey,
    newTypeKey,
    setNewTypeKey,
    newTypeDisplay,
    setNewTypeDisplay,
    quickCreateProfile,
    createProfileType,
    createNewThread,
  } = props;

  return (
    <section>
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Create New Thread</h2>
        <button
          type="button"
          className={`text-sm underline ${typesLoading ? "opacity-60 cursor-not-allowed" : ""}`}
          onClick={loadProfileTypesOrSeed}
          disabled={typesLoading}
          title="Fetch /profile-types; auto-seed defaults if empty"
        >
          {typesLoading ? "Loading types…" : "Load participant types"}
        </button>
      </div>
      <div className="mt-2 grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="border rounded-xl p-3 bg-white space-y-2">
          <label className="text-sm">Project ID</label>
          <input className="border rounded px-2 py-1 w-full" value={projectIdNew} onChange={(e) => setProjectIdNew(e.target.value)} placeholder="proj-123" />
          <label className="text-sm">Title</label>
          <input className="border rounded px-2 py-1 w-full" value={titleNew} onChange={(e) => setTitleNew(e.target.value)} placeholder="My Debate" />
          <label className="text-sm">Phase</label>
          <input className="border rounded px-2 py-1 w-full" value={phaseNew} onChange={(e) => setPhaseNew(e.target.value)} placeholder="phase1-exploration" />
        </div>
        <div className="border rounded-xl p-3 bg-white space-y-2">
          <div className="text-sm font-medium">Participants (type keys)</div>
          <div className="grid grid-cols-2 gap-2">
            {profileTypes.length === 0 && <div className="text-xs text-gray-500 col-span-2">Click "Load participant types".</div>}
            {profileTypes.map((pt: any) => (
              <label key={pt.id} className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={!!selectedTypeKeys[pt.key]} onChange={(e) => setSelectedTypeKeys((prev: any) => ({ ...prev, [pt.key]: e.target.checked }))} />
                <span>
                  <span className="font-mono text-xs">{pt.key}</span> · {pt.display_name}
                </span>
              </label>
            ))}
          </div>
          <div className="text-xs text-gray-500">Defaults after seeding: pm, frontend, backend, architect, qa, critic.</div>
        </div>
        <div className="border rounded-xl p-3 bg-white space-y-2">
          <label className="text-sm">Initial user message (optional)</label>
          <textarea className="border rounded px-2 py-1 w-full min-h-[60px]" value={initialMessageNew} onChange={(e) => setInitialMessageNew(e.target.value)} placeholder="Kick off the discussion…" />
          <label className="text-sm">Context JSON (optional)</label>
          <textarea className="border rounded px-2 py-1 w-full min-h-[60px] font-mono text-xs" value={contextNew} onChange={(e) => setContextNew(e.target.value)} placeholder='{"goals":["…"],"constraints":["…"]}' />
          <button className="px-3 py-1.5 rounded bg-emerald-600 text-white" onClick={createNewThread}>
            Create & Load
          </button>
        </div>
      </div>

      {/* Quick-create + Types mgmt */}
      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="border rounded-xl p-3 bg-white space-y-3">
          <div>
            <div className="font-medium text-sm">Quick-create a new profile</div>
            <div className="flex flex-wrap items-center gap-2 mt-1.5">
              <input className="border rounded px-2 py-1 text-sm" placeholder="Name (e.g., Data Scientist)" value={quickProfileName} onChange={(e) => setQuickProfileName(e.target.value)} />
              <select className="border rounded px-2 py-1 text-sm" value={quickProfileTypeKey} onChange={(e) => setQuickProfileTypeKey(e.target.value)}>
                <option value="">— pick type key —</option>
                {profileTypes.map((pt: any) => (
                  <option key={pt.id} value={pt.key}>
                    {pt.key} · {pt.display_name}
                  </option>
                ))}
              </select>
              <button className="px-3 py-1.5 rounded bg-emerald-600 text-white text-sm" onClick={() => quickCreateProfile(false)}>
                Create profile
              </button>
            </div>
            <div className="text-xs text-gray-500 mt-1">If the type doesn’t exist yet, create it below first.</div>
          </div>

          <div>
            <div className="font-medium text-sm">Create a new profile type</div>
            <div className="flex flex-wrap items-center gap-2 mt-1.5">
              <input className="border rounded px-2 py-1 text-sm font-mono" placeholder="type key (e.g., data_scientist)" value={newTypeKey} onChange={(e) => setNewTypeKey(e.target.value)} />
              <input className="border rounded px-2 py-1 text-sm" placeholder="Display name (e.g., Data Scientist)" value={newTypeDisplay} onChange={(e) => setNewTypeDisplay(e.target.value)} />
              <button className="px-3 py-1.5 rounded bg-gray-900 text-white text-sm" onClick={createProfileType}>
                Create type
              </button>
              <button className="px-3 py-1.5 rounded bg-gray-200 text-gray-900 text-sm border" onClick={loadProfileTypes}>
                Reload types
              </button>
            </div>
          </div>
        </div>

        <div className="border rounded-xl p-3 bg-white">
          <div className="text-xs text-gray-600">
            After creating the thread, switch to the <span className="font-medium">Participants</span> tab to add people by type or by existing profile.
          </div>
        </div>
      </div>
    </section>
  );
}

export function ParticipantsTab({
  thread,
  participantIds,
  participantMap,
  nextSpeakerId,
  selectedNextSpeakerId,
  setSelectedNextSpeakerId,
  applyNextSpeaker,
  profileTypes,
  addTypeKey,
  setAddTypeKey,
  addExistingProfileId,
  setAddExistingProfileId,
  profiles,
  addParticipantByType,
  addParticipantByProfile,
}: any) {
  return (
    <section>
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Participants</h2>
        <div className="text-sm text-gray-600">Phase: <span className="font-mono">{thread?.phase || "—"}</span></div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <div className="text-sm">
          Next now: <span className="font-medium">{participantMap[nextSpeakerId]?.name || (nextSpeakerId || "").slice(0, 6) || "—"}</span>
        </div>
        <select className="border rounded px-2 py-1 text-sm" value={selectedNextSpeakerId} onChange={(e) => setSelectedNextSpeakerId(e.target.value)}>
          {participantIds.map((id: string) => (
            <option key={id} value={id}>
              {participantMap[id]?.name || id.slice(0, 6)}
            </option>
          ))}
        </select>
        <button className="px-2.5 py-1.5 rounded bg-gray-800 text-white text-sm" onClick={applyNextSpeaker} disabled={!thread}>
          Apply
        </button>
      </div>

      <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
        {participantIds.map((id: string, idx: number) => {
          const p = participantMap[id];
          const isNext = id === nextSpeakerId;
          return (
            <div key={id} className={`border rounded-xl p-3 flex items-center justify-between ${isNext ? "bg-yellow-50 border-yellow-300" : "bg-white"}`}>
              <div>
                <div className="font-medium">{p?.name || id.slice(0, 6)}</div>
                <div className="text-xs text-gray-600">{p?.role || "agent"}</div>
              </div>
              <div className="text-xs text-gray-500">#{idx + 1}</div>
            </div>
          );
        })}
      </div>

      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="border rounded-xl p-3 bg-white space-y-2">
          <div className="font-medium text-sm">Add participant</div>
          <div className="flex flex-wrap items-center gap-2">
            <select className="border rounded px-2 py-1 text-sm" value={addTypeKey} onChange={(e) => setAddTypeKey(e.target.value)}>
              <option value="">— pick type key —</option>
              {profileTypes.map((pt: any) => (
                <option key={pt.id} value={pt.key}>
                  {pt.key} · {pt.display_name}
                </option>
              ))}
            </select>
            <button className="px-3 py-1.5 rounded bg-blue-600 text-white text-sm" onClick={addParticipantByType} disabled={!thread || !addTypeKey}>
              Add by type
            </button>

            <select className="border rounded px-2 py-1 text-sm" value={addExistingProfileId} onChange={(e) => setAddExistingProfileId(e.target.value)}>
              <option value="">— pick existing profile —</option>
              {Object.values(profiles).map((p: any) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {p.role}
                </option>
              ))}
            </select>
            <button className="px-3 py-1.5 rounded bg-indigo-600 text-white text-sm" onClick={addParticipantByProfile} disabled={!thread || !addExistingProfileId}>
              Add by profile
            </button>
          </div>
          <div className="text-xs text-gray-500">Tip: if your type key isn’t listed, create it in Create & Types.</div>
        </div>

        <div className="border rounded-xl p-3 bg-white">
          <div className="text-xs text-gray-600">Manage roles and participation via type keys and profiles.</div>
        </div>
      </div>
    </section>
  );
}

export function ArtifactsTab({ doCompileP1, doCompileP2, phase1, phase2, threadId }: any) {
  return (
    <section>
      <div className="border rounded-xl p-3 bg-white">
        <div className="font-medium">Compile</div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button className="px-3 py-1.5 rounded bg-purple-600 text-white" onClick={doCompileP1} disabled={!threadId}>
            Phase-1 artifact
          </button>
          <button className="px-3 py-1.5 rounded bg-purple-700 text-white" onClick={doCompileP2} disabled={!threadId}>
            Phase-2 reqs+tests
          </button>
        </div>
        {(phase1 || phase2) && (
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            {phase1 && <pre className="p-2 bg-gray-50 border rounded overflow-auto max-h-64">{JSON.stringify(phase1, null, 2)}</pre>}
            {phase2 && <pre className="p-2 bg-gray-50 border rounded overflow-auto max-h-64">{JSON.stringify(phase2, null, 2)}</pre>}
          </div>
        )}
      </div>
    </section>
  );
}

export function BrowseThreadsTab({ projectFilter, setProjectFilter, loadThreadsList, threadList, onPick }: any) {
  return (
    <section>
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Browse Threads</h2>
      </div>
      <div className="flex items-center gap-2 mt-2">
        <input className="border rounded px-2 py-1" placeholder="Filter by project_id" value={projectFilter} onChange={(e) => setProjectFilter(e.target.value)} />
        <button className="px-3 py-1.5 rounded bg-gray-800 text-white" onClick={loadThreadsList}>
          Load Threads
        </button>
      </div>
      {threadList.length > 0 && (
        <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
          {threadList.map((t: any) => (
            <button key={t.id} onClick={() => onPick(t)} className="text-left border rounded-xl p-3 bg-white hover:border-gray-400">
              <div className="font-medium">{t.title}</div>
              <div className="text-xs text-gray-600">id: <span className="font-mono">{t.id}</span></div>
              <div className="text-xs text-gray-600">
                project: <span className="font-mono">{t.project_id}</span> · phase: <span className="font-mono">{t.phase}</span>
              </div>
              <div className="text-xs text-gray-600">participants: {t.participants?.length ?? 0}</div>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

export function PrivateTab({
  participantIds,
  participantMap,
  adviceParticipantId,
  setAdviceParticipantId,
  adviceQuestion,
  setAdviceQuestion,
  createPrivateThread,
  privateThreads,
  activePrivateThreadId,
  setActivePrivateThreadId,
  loadPrivateThreadById,
  privateChrono,
  privateComposer,
  setPrivateComposer,
  sendPrivateMessage,
  stepPrivateOnce,
}: any) {
  return (
    <section className="space-y-3">
      <div className="border rounded-xl p-3 bg-white space-y-2">
        <div className="font-medium text-sm">Ask private advice (off-record)</div>
        <div className="flex flex-wrap items-center gap-2">
          <select className="border rounded px-2 py-1 text-sm" value={adviceParticipantId} onChange={(e) => setAdviceParticipantId(e.target.value)}>
            <option value="">— pick participant —</option>
            {participantIds.map((id: string) => (
              <option key={id} value={id}>
                {participantMap[id]?.name || id.slice(0, 6)} · {participantMap[id]?.role}
              </option>
            ))}
          </select>
        </div>
        <textarea
          className="border rounded px-2 py-1 w-full min-h-[60px] text-sm"
          placeholder="Your private question for this participant (e.g., PM: who else should we invite?)"
          value={adviceQuestion}
          onChange={(e) => setAdviceQuestion(e.target.value)}
        />
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 rounded bg-teal-700 text-white text-sm" onClick={() => createPrivateThread(adviceParticipantId, adviceQuestion)} disabled={!adviceParticipantId || !adviceQuestion.trim()}>
            Ask privately
          </button>
          <span className="text-xs text-gray-500">We’ll create a private thread and auto-run one turn.</span>
        </div>
      </div>

      {privateThreads.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <div className="border rounded-xl p-3 bg-white">
            <div className="text-sm font-medium mb-2">Your private consults</div>
            <ul className="space-y-1">
              {privateThreads.map((pt: any) => (
                <li key={pt.id}>
                  <button
                    className={`text-left w-full border rounded px-2 py-1 text-sm ${activePrivateThreadId === pt.id ? "bg-gray-100" : "bg-white"}`}
                    onClick={async () => {
                      setActivePrivateThreadId(pt.id);
                      await loadPrivateThreadById(pt.id);
                    }}
                  >
                    <div className="font-medium">{pt.title}</div>
                    <div className="text-xs text-gray-600 font-mono">{pt.id}</div>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <div className="border rounded-xl p-3 bg-white md:col-span-2">
            {activePrivateThreadId ? (
              <>
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">Thread: <span className="font-mono">{activePrivateThreadId}</span></div>
                  <div className="flex items-center gap-2">
                    <button className="px-3 py-1.5 rounded bg-blue-700 text-white text-sm" onClick={() => stepPrivateOnce(activePrivateThreadId)}>
                      Step private once
                    </button>
                    <button className="px-3 py-1.5 rounded bg-gray-100 text-gray-900 border text-sm" onClick={() => loadPrivateThreadById(activePrivateThreadId)}>
                      Refresh
                    </button>
                  </div>
                </div>
                <div className="mt-2 border rounded bg-white">
                  <div className="p-3 h-[280px] overflow-auto space-y-3">
                    {privateChrono.map((m: any, i: number) => {
                      const isUser = m.sender_type === "user";
                      const who = isUser ? "You" : "Agent";
                      let text = "";
                      let proposals, questions;
                      if (typeof m.content === "object" && m.content !== null) {
                        text = String(m.content.message ?? "");
                        proposals = Array.isArray(m.content.proposals) ? m.content.proposals : undefined;
                        questions = Array.isArray(m.content.questions) ? m.content.questions : undefined;
                      } else {
                        text = String(m.content || "");
                      }
                      return (
                        <div key={m.id || i} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                          <div className={`max-w-[80%] rounded-2xl px-3 py-2 border ${isUser ? "bg-blue-50 border-blue-200" : "bg-gray-50"}`}>
                            <div className="text-xs text-gray-500 mb-1">{who}</div>
                            <div className="whitespace-pre-wrap">{text}</div>
                            {proposals?.length > 0 && (
                              <ul className="mt-2 list-disc pl-5 text-sm text-gray-700">
                                {proposals.map((p: any, idx: number) => (
                                  <li key={idx}>Proposal: {p}</li>
                                ))}
                              </ul>
                            )}
                            {questions?.length > 0 && (
                              <ul className="mt-2 list-disc pl-5 text-sm text-gray-700">
                                {questions.map((q: any, idx: number) => (
                                  <li key={idx}>Question: {q}</li>
                                ))}
                              </ul>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="border-t p-3 flex items-center gap-2">
                    <textarea
                      className="flex-1 border rounded-lg px-3 py-2 min-h-[44px] text-sm"
                      placeholder="Private message…"
                      value={privateComposer}
                      onChange={(e) => setPrivateComposer(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) sendPrivateMessage();
                      }}
                    />
                    <button className="px-4 py-2 rounded bg-gray-900 text-white text-sm" onClick={sendPrivateMessage} disabled={!privateComposer.trim()}>
                      Send
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-sm text-gray-600">Pick a private thread on the left.</div>
            )}
          </div>
        </div>
      ) : (
        <div className="text-sm text-gray-600">No private threads yet. Ask a question above.</div>
      )}
    </section>
  );
}

// -----------------
// Right pane (persistent)
// -----------------
export function RunPanel({
  stepMode,
  setStepMode,
  threadId,
  stepOnce,
  autorun,
  setAutorun,
  delayMs,
  setDelayMs,
  nSteps,
  setNSteps,
  runNSteps,
  refreshAll,
  loadPreview,
  loadDecisions,
  preview,
  decisions,
}: any) {
  return (
    <section className="border rounded-xl p-3 bg-white">
      <h2 className="font-semibold">Run</h2>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <select className="border rounded px-2 py-1" value={stepMode} onChange={(e) => setStepMode(e.target.value)}>
          <option value="mem">Step with memory</option>
          <option value="plain">Plain step</option>
        </select>
        <button className="px-3 py-1.5 rounded bg-blue-600 text-white hover:opacity-90" onClick={stepOnce} disabled={!threadId} title="Shortcut: Space">
          Step once
        </button>
        <div className="flex items-center gap-2">
          <label className="text-sm">Auto-run</label>
          <button
            className={`px-3 py-1.5 rounded ${autorun ? "bg-green-600" : "bg-gray-700"} text-white hover:opacity-90`}
            onClick={() => setAutorun((x: boolean) => !x)}
            disabled={!threadId}
            title="Shortcut: A"
          >
            {autorun ? "On" : "Off"}
          </button>
          <input className="border rounded px-2 py-1 w-24" type="number" min={250} step={250} value={delayMs} onChange={(e) => setDelayMs(Number(e.target.value) || 1000)} />
          <span className="text-sm text-gray-600">ms delay</span>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <input className="border rounded px-2 py-1 w-24" type="number" min={1} value={nSteps} onChange={(e) => setNSteps(Number(e.target.value) || 1)} />
        <button className="px-3 py-1.5 rounded bg-indigo-600 text-white hover:opacity-90" onClick={runNSteps} disabled={!threadId}>
          Run N steps
        </button>
        <button className="px-3 py-1.5 rounded bg-gray-100 text-gray-900 border" onClick={() => refreshAll()} disabled={!threadId}>
          Refresh
        </button>
        <button className="px-3 py-1.5 rounded bg-amber-600 text-white" onClick={loadPreview} disabled={!threadId}>
          Preview next turn
        </button>
        <button className="px-3 py-1.5 rounded bg-teal-700 text-white" onClick={loadDecisions} disabled={!threadId}>
          Load decisions
        </button>
      </div>

      {(preview || (decisions && decisions.length > 0)) && (
        <div className="mt-3 grid grid-cols-1 gap-3">
          {preview && (
            <div className="border rounded-xl bg-gray-50 p-3">
              <div className="font-medium">Step Preview</div>
              <div className="mt-2 grid grid-cols-1 gap-3 text-xs">
                <div>
                  <div className="text-gray-600 mb-1">System Prompt</div>
                  <pre className="p-2 bg-white border rounded overflow-auto max-h-48">{preview.system_prompt}</pre>
                </div>
                <div>
                  <div className="text-gray-600 mb-1">User Prompt</div>
                  <pre className="p-2 bg-white border rounded overflow-auto max-h-48">{preview.user_prompt}</pre>
                </div>
                <div>
                  <div className="text-gray-600 mb-1">Schema</div>
                  <pre className="p-2 bg-white border rounded overflow-auto max-h-48">{JSON.stringify(preview.schema, null, 2)}</pre>
                </div>
                {preview.memory_snippets?.length > 0 && (
                  <div>
                    <div className="text-gray-600 mb-1">Retrieved snippets</div>
                    <ul className="list-disc pl-5">
                      {preview.memory_snippets.map((s: string, i: number) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {decisions?.length > 0 && (
            <div className="border rounded-xl bg-gray-50 p-3">
              <div className="font-medium">Decisions (proposed)</div>
              <ul className="mt-2 space-y-1 text-sm">
                {decisions.map((d: any) => (
                  <li key={d.id} className="border rounded p-2 bg-white">
                    <div className="flex items-center justify-between">
                      <span>{d.text}</span>
                      <span className="text-xs px-2 py-0.5 rounded bg-gray-200">{d.status}</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">id: <span className="font-mono">{d.id}</span></div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

export function TranscriptPanel({ messages, fmtMessage, composer, setComposer, sendUserMessage, threadId }: any) {
  return (
    <section className="border rounded-xl bg-white">
      <div className="p-3">
        <h2 className="font-semibold">Transcript</h2>
      </div>
      <div className="px-3 pb-3">
        <div className="h-[360px] overflow-auto space-y-3">
          {[...messages].reverse().map((m: any, i: number) => {
            const fm = fmtMessage(m);
            const isUser = m.sender_type === "user";
            return (
              <div key={m.id || i} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[80%] rounded-2xl px-3 py-2 border ${isUser ? "bg-blue-50 border-blue-200" : "bg-gray-50"}`}>
                  <div className="text-xs text-gray-500 mb-1">{isUser ? fm.who : `${fm.who}${fm.role ? ` · ${fm.role}` : ""}`}</div>
                  <div className="whitespace-pre-wrap">{fm.text}</div>
                  {fm.proposals?.length > 0 && (
                    <ul className="mt-2 list-disc pl-5 text-sm text-gray-700">
                      {fm.proposals.map((p: any, idx: number) => (
                        <li key={idx}>Proposal: {p}</li>
                      ))}
                    </ul>
                  )}
                  {fm.questions?.length > 0 && (
                    <ul className="mt-2 list-disc pl-5 text-sm text-gray-700">
                      {fm.questions.map((q: any, idx: number) => (
                        <li key={idx}>Question: {q}</li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div className="border-t mt-3 pt-3 flex items-center gap-2">
          <textarea
            className="flex-1 border rounded-lg px-3 py-2 min-h-[44px]"
            placeholder="Type your message…"
            value={composer}
            onChange={(e) => setComposer(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) sendUserMessage();
            }}
          />
          <button className="px-4 py-2 rounded bg-gray-900 text-white" onClick={sendUserMessage} disabled={!threadId || !composer.trim()}>
            Send
          </button>
        </div>
      </div>
    </section>
  );
}
