import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  HeaderBar,
  TabBar,
  RunPanel,
  TranscriptPanel,
  CreateTypesTab,
  ParticipantsTab,
  ArtifactsTab,
  BrowseThreadsTab,
  PrivateTab,
} from "./DebatePanels";

// Simple runtime error boundary so a bug doesn't render a blank page
class ErrorBoundary extends React.Component<any, { hasError: boolean; msg?: string }> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(error: any) {
    return { hasError: true, msg: String(error?.message || error) };
  }
  componentDidCatch(error: any, info: any) {
    console.error("UI crashed:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="m-4 p-4 border rounded bg-red-50 text-red-800">
          <div className="font-semibold">Something went wrong rendering the UI.</div>
          <div className="text-sm mt-1">{this.state.msg}</div>
          <div className="text-xs mt-2 text-red-700">Check the browser console for stack traces.</div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function DebateControlPanel() {
  // -----------------
  // Config
  // -----------------
  const [baseUrl, setBaseUrl] = useState("http://localhost:8000/debate");
  const [threadId, setThreadId] = useState("");

  // -----------------
  // State
  // -----------------
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [thread, setThread] = useState<any>(null);
  const [profiles, setProfiles] = useState<Record<string, any>>({}); // id -> {id,name,role}
  const [messages, setMessages] = useState<any[]>([]); // newest-first from API

  const [autorun, setAutorun] = useState(false);
  const [delayMs, setDelayMs] = useState(1500);
  const [stepMode, setStepMode] = useState<"mem" | "plain">("mem");
  const [nSteps, setNSteps] = useState(3);

  const [composer, setComposer] = useState("");
  const [phase1, setPhase1] = useState<any>(null);
  const [phase2, setPhase2] = useState<any>(null);

  // Optional preview + decisions
  const [preview, setPreview] = useState<any>(null);
  const [decisions, setDecisions] = useState<any[]>([]);

  // Types + seeding
  const [typesLoading, setTypesLoading] = useState(false);
  const [typesSeededInfo, setTypesSeededInfo] = useState("");
  const [profileTypes, setProfileTypes] = useState<any[]>([]); // [{id,key,display_name}]

  // New thread creation UI
  const [projectIdNew, setProjectIdNew] = useState("");
  const [titleNew, setTitleNew] = useState("");
  const [phaseNew, setPhaseNew] = useState("phase1-exploration");
  const [initialMessageNew, setInitialMessageNew] = useState("");
  const [contextNew, setContextNew] = useState("{}");
  const [selectedTypeKeys, setSelectedTypeKeys] = useState<Record<string, boolean>>({});

  // Thread browser
  const [projectFilter, setProjectFilter] = useState("");
  const [threadList, setThreadList] = useState<any[]>([]);

  // Participants helpers
  const [addTypeKey, setAddTypeKey] = useState("");
  const [addExistingProfileId, setAddExistingProfileId] = useState("");
  const [quickProfileName, setQuickProfileName] = useState("");
  const [quickProfileTypeKey, setQuickProfileTypeKey] = useState("");
  const [newTypeKey, setNewTypeKey] = useState("");
  const [newTypeDisplay, setNewTypeDisplay] = useState("");

  // Next speaker (selected)
  const [selectedNextSpeakerId, setSelectedNextSpeakerId] = useState("");

  // Private advice
  const [adviceParticipantId, setAdviceParticipantId] = useState("");
  const [adviceQuestion, setAdviceQuestion] = useState("");
  const [privateThreads, setPrivateThreads] = useState<any[]>([]); // [{id,title}]
  const [activePrivateThreadId, setActivePrivateThreadId] = useState("");
  const [privateMessages, setPrivateMessages] = useState<Record<string, any[]>>({}); // id -> messages (newest-first)
  const [privateComposer, setPrivateComposer] = useState("");

  const intervalRef = useRef<any>(null);

  // -----------------
  // Tabs (left column)
  // -----------------
  const TAB_ITEMS = [
    { key: "create", label: "Create & Types" },
    { key: "participants", label: "Participants" },
    { key: "artifacts", label: "Artifacts" },
    { key: "threads", label: "Browse" },
    { key: "private", label: "Private" },
  ];

  const [activeTab, setActiveTab] = useState("create");
  // hydrate active tab from hash safely on client only
  useEffect(() => {
    if (typeof window === "undefined") return;
    const qs = new URLSearchParams(window.location.hash.slice(1));
    const initial = qs.get("tab");
    if (initial) setActiveTab(initial);
  }, []);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const qs = new URLSearchParams(window.location.hash.slice(1));
    qs.set("tab", activeTab);
    window.location.hash = qs.toString();
  }, [activeTab]);

  // -----------------
  // API endpoints
  // -----------------
  const api = useMemo(() => {
    const safeBase = baseUrl.replace(/\/$/, "");
    return {
      profiles: `${safeBase}/profiles`,
      profileTypes: `${safeBase}/profile-types`,
      seedProfiles: `${safeBase}/profiles/seed`,
      createProfile: `${safeBase}/profiles`,
      createProfileType: `${safeBase}/profile-types`,
      addParticipant: (id: string) => `${safeBase}/threads/${id}/participants`,
      setCursor: (id: string) => `${safeBase}/threads/${id}/cursor`,
      thread: (id: string) => `${safeBase}/threads/${id}`,
      threadsList: (project?: string) =>
        project ? `${safeBase}/threads?project_id=${encodeURIComponent(project)}` : `${safeBase}/threads`,
      messages: (id: string) => `${safeBase}/threads/${id}/messages`,
      postMessage: (id: string) => `${safeBase}/threads/${id}/messages`,
      step: (id: string) => `${safeBase}/threads/${id}/${stepMode === "mem" ? "step_mem" : "step"}`,
      autorun: (id: string, n: number) => `${safeBase}/threads/${id}/autorun?n=${n}`,
      compileP1: (id: string) => `${safeBase}/threads/${id}/compile/phase1`,
      compileP2: (id: string) => `${safeBase}/threads/${id}/compile/phase2`,
      createThreadNew: (projectId: string) => `${safeBase}/projects/${projectId}/threads/new`,
      stepPreview: (id: string) => `${safeBase}/threads/${id}/step_preview`,
      decisions: (id: string) => `${safeBase}/threads/${id}/decisions`,
    };
  }, [baseUrl, stepMode]);

  // -----------------
  // Loaders
  // -----------------
  const loadProfiles = useCallback(async () => {
    const r = await fetch(api.profiles);
    if (!r.ok) throw new Error(`profiles: ${r.status}`);
    const data = await r.json();
    const map: Record<string, any> = {};
    for (const p of data) map[p.id] = p;
    setProfiles(map);
  }, [api.profiles]);

  const loadProfileTypes = useCallback(async () => {
    const r = await fetch(api.profileTypes);
    if (!r.ok) throw new Error(`profile-types: ${r.status}`);
    const data = await r.json();
    setProfileTypes(data);
    return data;
  }, [api.profileTypes]);

  const loadProfileTypesOrSeed = useCallback(async () => {
    setTypesLoading(true);
    setTypesSeededInfo("");
    setError(null);
    try {
      let types = await loadProfileTypes();
      if (!types || types.length === 0) {
        const seedResp = await fetch(api.seedProfiles, { method: "POST" });
        if (!seedResp.ok) {
          const t = await seedResp.text().catch(() => "");
          throw new Error(`seeding defaults failed: ${seedResp.status}${t ? ` ${t}` : ""}`);
        }
        const seed = await seedResp.json().catch(() => ({}));
        setTypesSeededInfo(`Seeded default profiles (inserted: ${seed?.inserted ?? "?"}).`);
        types = await loadProfileTypes();
        await loadProfiles();
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setTypesLoading(false);
    }
  }, [api.seedProfiles, loadProfileTypes, loadProfiles]);

  const loadThread = useCallback(async (id: string) => {
    const r = await fetch(api.thread(id));
    if (!r.ok) throw new Error(`thread ${id}: ${r.status}`);
    const data = await r.json();
    setThread(data);
    const participants = data?.participants || [];
    if (participants.length > 0) {
      const idx = data.cursor % (participants.length || 1);
      setSelectedNextSpeakerId(participants[idx]);
    } else {
      setSelectedNextSpeakerId("");
    }
  }, [api]);

  const loadMessages = useCallback(async (id: string) => {
    const r = await fetch(api.messages(id));
    if (!r.ok) throw new Error(`messages ${id}: ${r.status}`);
    const data = await r.json();
    setMessages(data); // newest-first
  }, [api]);

  const loadThreadsList = useCallback(async () => {
    const r = await fetch(api.threadsList(projectFilter || undefined));
    if (!r.ok) throw new Error(`threads list: ${r.status}`);
    const data = await r.json();
    setThreadList(data);
  }, [api, projectFilter]);

  const loadPreview = useCallback(async () => {
    if (!threadId) return;
    setError(null);
    try {
      const r = await fetch(api.stepPreview(threadId));
      if (!r.ok) throw new Error(`preview: ${r.status}`);
      const data = await r.json();
      setPreview(data);
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  }, [threadId, api]);

  const loadDecisions = useCallback(async () => {
    if (!threadId) return;
    setError(null);
    try {
      const r = await fetch(api.decisions(threadId));
      if (!r.ok) throw new Error(`decisions: ${r.status}`);
      const data = await r.json();
      setDecisions(data);
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  }, [threadId, api]);

  const refreshAll = useCallback(async () => {
    if (!threadId) return;
    setError(null);
    try {
      setLoading(true);
      await Promise.all([loadProfiles(), loadThread(threadId), loadMessages(threadId), loadDecisions()]);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [threadId, loadProfiles, loadThread, loadMessages, loadDecisions]);

  // -----------------
  // Autorun loop
  // -----------------
  useEffect(() => {
    if (!autorun) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
      return;
    }
    if (!threadId) return;

    let busy = false;
    intervalRef.current = setInterval(async () => {
      if (busy) return;
      busy = true;
      try {
        await stepOnce();
      } catch (e: any) {
        console.error(e);
        setError(e?.message || String(e));
        setAutorun(false);
      } finally {
        busy = false;
      }
    }, Math.max(250, Number(delayMs) || 1000));

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [autorun, delayMs, threadId, stepMode]);

  // -----------------
  // Keyboard shortcuts
  // -----------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && t.tagName === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        stepOnce();
      } else if (e.key && e.key.toLowerCase() === "a") {
        setAutorun((x) => !x);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [threadId, stepMode]);

  // -----------------
  // Actions
  // -----------------
  const stepOnce = useCallback(async () => {
    if (!threadId) return;
    setError(null);
    const r = await fetch(api.step(threadId), { method: "POST" });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`step failed: ${r.status}${text ? ` ${text}` : ""}`);
    }
    await loadThread(threadId);
    await loadMessages(threadId);
    await loadDecisions();
    setPreview(null);
  }, [threadId, api, loadThread, loadMessages, loadDecisions]);

  const runNSteps = useCallback(async () => {
    if (!threadId) return;
    setError(null);
    const r = await fetch(api.autorun(threadId, Math.max(1, Number(nSteps) || 1)), { method: "POST" });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`autorun failed: ${r.status}${text ? ` ${text}` : ""}`);
    }
    await loadThread(threadId);
    await loadMessages(threadId);
    await loadDecisions();
    setPreview(null);
  }, [threadId, api, nSteps, loadThread, loadMessages, loadDecisions]);

  const sendUserMessage = useCallback(async () => {
    if (!composer.trim() || !threadId) return;
    setError(null);
    const r = await fetch(api.postMessage(threadId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: composer.trim(), content_type: "text" }),
    });
    if (!r.ok) throw new Error(`post message failed: ${r.status}`);
    setComposer("");
    await loadThread(threadId);
    await loadMessages(threadId);
  }, [composer, threadId, api, loadThread, loadMessages]);

  const doCompileP1 = useCallback(async () => {
    if (!threadId) return;
    setError(null);
    const r = await fetch(api.compileP1(threadId), { method: "POST" });
    if (!r.ok) throw new Error(`compile p1: ${r.status}`);
    const data = await r.json();
    setPhase1(data.artifact);
  }, [threadId, api]);

  const doCompileP2 = useCallback(async () => {
    if (!threadId) return;
    setError(null);
    const r = await fetch(api.compileP2(threadId), { method: "POST" });
    if (!r.ok) throw new Error(`compile p2: ${r.status}`);
    const data = await r.json();
    setPhase2(data);
  }, [threadId, api]);

  const createNewThread = useCallback(async () => {
    if (!projectIdNew || !titleNew) {
      setError("Project ID and Title are required to create a thread.");
      return;
    }
    const keys = Object.keys(selectedTypeKeys).filter((k) => selectedTypeKeys[k]);
    if (keys.length === 0) {
      setError("Select at least one participant type.");
      return;
    }
    let ctx: any;
    const trimmed = (contextNew || "").trim();
    if (trimmed) {
      try {
        ctx = JSON.parse(trimmed);
      } catch {
        setError("Context must be valid JSON.");
        return;
      }
    }
    setError(null);
    const payload: any = {
      phase: phaseNew,
      title: titleNew,
      participant_type_keys: keys,
      context: ctx,
      initial_message: initialMessageNew || undefined,
    };
    const r = await fetch(api.createThreadNew(projectIdNew), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(`create thread failed: ${r.status} ${text}`);
    }
    const data = await r.json();
    setThreadId(data.id);
    setThread(data);
    setPreview(null);
    setDecisions([]);
    await Promise.all([loadProfiles(), loadThread(data.id), loadMessages(data.id)]);
  }, [api, projectIdNew, titleNew, phaseNew, contextNew, initialMessageNew, selectedTypeKeys, loadProfiles, loadThread, loadMessages]);

  // ---- Private advice helpers ----
  const chronologicalMessages = useMemo(() => [...messages].reverse(), [messages]);
  const buildContextSnippet = useCallback(() => {
    const recent = chronologicalMessages.slice(-8);
    const lines = recent.map((m: any) => {
      const isUser = m.sender_type === "user";
      const who = isUser ? "USER" : profiles[m.sender_agent_id]?.name || "AGENT";
      let text = "";
      if (typeof m.content === "object" && m.content !== null) {
        text = (m.content as any).message ?? JSON.stringify(m.content);
      } else {
        text = String(m.content || "");
      }
      return `${who}: ${text}`;
    });
    return lines.join("\n");
  }, [chronologicalMessages, profiles]);

  const loadPrivateThreadById = useCallback(async (id: string) => {
    try {
      const rt = await fetch(api.thread(id));
      if (rt.ok) {
        const tdata = await rt.json();
        setPrivateThreads((arr) => {
          const seen = arr.find((x: any) => x.id === id);
          if (seen) return arr.map((x: any) => (x.id === id ? { id, title: tdata.title } : x));
          return [{ id, title: tdata.title }, ...arr];
        });
      }
    } catch (_) {}
    try {
      const rm = await fetch(api.messages(id));
      if (rm.ok) {
        const msgData = await rm.json();
        setPrivateMessages((m) => ({ ...m, [id]: msgData }));
      }
    } catch (_) {}
  }, [api]);

  const stepPrivateOnce = useCallback(async (id: string) => {
    setError(null);
    const r = await fetch(api.step(id), { method: "POST" });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`private step failed: ${r.status}${text ? ` ${text}` : ""}`);
    }
    await loadPrivateThreadById(id);
  }, [api, loadPrivateThreadById]);

  const sendPrivateMessage = useCallback(async () => {
    if (!activePrivateThreadId || !privateComposer.trim()) return;
    setError(null);
    const r = await fetch(api.postMessage(activePrivateThreadId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: privateComposer.trim(), content_type: "text" }),
    });
    if (!r.ok) throw new Error(`post private message failed: ${r.status}`);
    setPrivateComposer("");
    await loadPrivateThreadById(activePrivateThreadId);
  }, [api, activePrivateThreadId, privateComposer, loadPrivateThreadById]);

  const createPrivateThread = useCallback(
    async (participantId: string, question: string) => {
      if (!thread || !participantId || !question.trim()) {
        setError("Private consult: pick a participant and write a question.");
        return null;
      }
      setError(null);
      const pMeta = profiles[participantId];
      const title = `Private: ${(pMeta?.name || participantId.slice(0, 6))} · ${thread.title}`;
      const payload = {
        phase: `${thread.phase}-advice`,
        title,
        participant_ids: [participantId],
        context: {
          off_record: true,
          source_thread_id: thread.id,
          source_phase: thread.phase,
          source_title: thread.title,
          source_project_id: thread.project_id,
        },
        initial_message:
          `OFF-RECORD ADVICE REQUEST for ${pMeta?.name || participantId}.\n` +
          `Please answer succinctly. Your answer will not be logged to the main thread.\n\n` +
          `QUESTION:\n${question.trim()}\n\n` +
          `SNIPPET FROM SOURCE THREAD:\n${buildContextSnippet()}`,
      } as any;

      const r = await fetch(api.createThreadNew(thread.project_id), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const t = await r.text().catch(() => "");
        throw new Error(`create private thread failed: ${r.status}${t ? ` ${t}` : ""}`);
      }
      const tdata = await r.json();
      setPrivateThreads((arr) => [{ id: tdata.id, title: tdata.title }, ...arr]);
      setActivePrivateThreadId(tdata.id);

      await loadPrivateThreadById(tdata.id);
      try {
        await stepPrivateOnce(tdata.id);
      } catch (e: any) {
        setError(e?.message || String(e));
      }

      return tdata.id;
    },
    [api, thread, profiles, buildContextSnippet, loadPrivateThreadById, stepPrivateOnce]
  );

  // Derived
  const participantIds = useMemo(() => thread?.participants || [], [thread]);
  const participantMap = useMemo(() => {
    const out: Record<string, { name: string; role: string }> = {};
    for (const id of participantIds) {
      const p = profiles[id];
      if (p) out[id] = { name: p.name, role: p.role };
    }
    return out;
  }, [participantIds, profiles]);

  const nextSpeakerIdx = thread ? thread.cursor % (participantIds.length || 1) : 0;
  const nextSpeakerId = participantIds[nextSpeakerIdx];

  function fmtMessage(m: any) {
    if (m.sender_type === "user") {
      return { who: "You", text: typeof m.content === "string" ? m.content : JSON.stringify(m.content) };
    }
    const meta = participantMap[m.sender_agent_id] || { name: (m.sender_agent_id || "").slice(0, 6) || "Agent", role: "" };
    let text = "";
    let proposals: string[] | undefined;
    let questions: string[] | undefined;
    if (typeof m.content === "object" && m.content !== null) {
      text = String(m.content.message ?? "");
      proposals = Array.isArray(m.content.proposals) ? m.content.proposals : undefined;
      questions = Array.isArray(m.content.questions) ? m.content.questions : undefined;
    } else {
      text = String(m.content);
    }
    return { who: meta.name, role: meta.role, text, proposals, questions };
  }

  const privateChrono = useMemo(() => {
    const arr = privateMessages[activePrivateThreadId] || [];
    return [...arr].reverse();
  }, [privateMessages, activePrivateThreadId]);

  // -----------------
  // UI — Two-column layout with persistent right pane
  // -----------------
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        {/* Header */}
        <div className="sticky top-0 z-20 border-b bg-white/80 backdrop-blur">
          <div className="max-w-7xl mx-auto p-4 sm:p-6">
            <HeaderBar
              baseUrl={baseUrl}
              setBaseUrl={setBaseUrl}
              threadId={threadId}
              setThreadId={setThreadId}
              loading={loading}
              onLoad={refreshAll}
            />
            <div className="mt-3 text-xs text-gray-500">
              Shortcuts — <kbd className="px-1 border rounded">Space</kbd> step once, <kbd className="px-1 border rounded">A</kbd> toggle auto-run.
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* LEFT: TABS CONTENT */}
          <div className="lg:col-span-7">
            {error && <div className="mb-3 p-3 rounded bg-red-100 text-red-800 text-sm">{String(error)}</div>}
            {typesSeededInfo && (
              <div className="mb-3 p-2 rounded bg-emerald-50 text-emerald-800 text-xs">{typesSeededInfo}</div>
            )}

            <TabBar items={TAB_ITEMS} active={activeTab} onChange={setActiveTab} />

            {activeTab === "create" && (
              <CreateTypesTab
                profileTypes={profileTypes}
                loadProfileTypes={loadProfileTypes}
                loadProfileTypesOrSeed={loadProfileTypesOrSeed}
                typesLoading={typesLoading}
                projectIdNew={projectIdNew}
                setProjectIdNew={setProjectIdNew}
                titleNew={titleNew}
                setTitleNew={setTitleNew}
                phaseNew={phaseNew}
                setPhaseNew={setPhaseNew}
                initialMessageNew={initialMessageNew}
                setInitialMessageNew={setInitialMessageNew}
                contextNew={contextNew}
                setContextNew={setContextNew}
                selectedTypeKeys={selectedTypeKeys}
                setSelectedTypeKeys={setSelectedTypeKeys}
                quickProfileName={quickProfileName}
                setQuickProfileName={setQuickProfileName}
                quickProfileTypeKey={quickProfileTypeKey}
                setQuickProfileTypeKey={setQuickProfileTypeKey}
                newTypeKey={newTypeKey}
                setNewTypeKey={setNewTypeKey}
                newTypeDisplay={newTypeDisplay}
                setNewTypeDisplay={setNewTypeDisplay}
                quickCreateProfile={quickCreateProfile}
                createProfileType={createProfileType}
                createNewThread={createNewThread}
              />
            )}

            {activeTab === "participants" && (
              <ParticipantsTab
                thread={thread}
                participantIds={participantIds}
                participantMap={participantMap}
                nextSpeakerId={nextSpeakerId}
                selectedNextSpeakerId={selectedNextSpeakerId}
                setSelectedNextSpeakerId={setSelectedNextSpeakerId}
                applyNextSpeaker={applyNextSpeaker}
                profileTypes={profileTypes}
                addTypeKey={addTypeKey}
                setAddTypeKey={setAddTypeKey}
                addExistingProfileId={addExistingProfileId}
                setAddExistingProfileId={setAddExistingProfileId}
                profiles={profiles}
                addParticipantByType={addParticipantByType}
                addParticipantByProfile={addParticipantByProfile}
              />
            )}

            {activeTab === "artifacts" && (
              <ArtifactsTab doCompileP1={doCompileP1} doCompileP2={doCompileP2} phase1={phase1} phase2={phase2} threadId={threadId} />
            )}

            {activeTab === "threads" && (
              <BrowseThreadsTab
                projectFilter={projectFilter}
                setProjectFilter={setProjectFilter}
                loadThreadsList={loadThreadsList}
                threadList={threadList}
                onPick={async (t: any) => {
                  setThreadId(t.id);
                  setThread(t);
                  setPreview(null);
                  setDecisions([]);
                  await loadProfiles();
                  await loadMessages(t.id);
                }}
              />
            )}

            {activeTab === "private" && (
              <PrivateTab
                participantIds={participantIds}
                participantMap={participantMap}
                adviceParticipantId={adviceParticipantId}
                setAdviceParticipantId={setAdviceParticipantId}
                adviceQuestion={adviceQuestion}
                setAdviceQuestion={setAdviceQuestion}
                createPrivateThread={createPrivateThread}
                privateThreads={privateThreads}
                activePrivateThreadId={activePrivateThreadId}
                setActivePrivateThreadId={setActivePrivateThreadId}
                loadPrivateThreadById={loadPrivateThreadById}
                privateChrono={privateChrono}
                privateComposer={privateComposer}
                setPrivateComposer={setPrivateComposer}
                sendPrivateMessage={sendPrivateMessage}
                stepPrivateOnce={stepPrivateOnce}
              />
            )}
          </div>

          {/* RIGHT: PERSISTENT PANELS */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="sticky top-[88px]">
              <RunPanel
                stepMode={stepMode}
                setStepMode={setStepMode}
                threadId={threadId}
                stepOnce={stepOnce}
                autorun={autorun}
                setAutorun={setAutorun}
                delayMs={delayMs}
                setDelayMs={setDelayMs}
                nSteps={nSteps}
                setNSteps={setNSteps}
                runNSteps={runNSteps}
                refreshAll={refreshAll}
                loadPreview={loadPreview}
                loadDecisions={loadDecisions}
                preview={preview}
                decisions={decisions}
              />

              <div className="mt-4">
                <TranscriptPanel
                  messages={messages}
                  fmtMessage={fmtMessage}
                  composer={composer}
                  setComposer={setComposer}
                  sendUserMessage={sendUserMessage}
                  threadId={threadId}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}

// Participant ops
async function addParticipantByType(this: any): Promise<void> {}
async function addParticipantByProfile(this: any): Promise<void> {}
async function applyNextSpeaker(this: any): Promise<void> {}
async function quickCreateProfile(this: any): Promise<void> {}
async function createProfileType(this: any): Promise<void> {}
