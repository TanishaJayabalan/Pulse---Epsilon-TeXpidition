import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BadgeCheck,
  BrainCircuit,
  Check,
  Clock,
  Database,
  Gauge,
  Layers,
  PauseCircle,
  RefreshCw,
  Send,
  SlidersHorizontal,
  UserRound,
  X,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";
import { Badge, Button, Card, GhostButton, Input, Select, Textarea } from "./components/ui";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8003";

type Interest = { category: string; entity: string; weight: number };

type TopProduct = {
  product_id: string;
  name: string;
  category: string;
  combined_score: number;
  similarity: number;
  intent_score: number;
};

type Recommendation = {
  id: string;
  customer_id: string;
  customer_name: string;
  brand: string;
  action: string;
  channel: string;
  timing: string;
  product: string;
  intent_score: number;
  fatigue_score: number;
  status: string;
  diagnosis: string;
  fatigue_cause: string;
  explanation: string;
  top_products: TopProduct[];
  suppression: boolean;
  decision_source: string;
  nba_action: string;
  nba_strategy?: string;
  nba_reason?: string;
  model_version: string;
};

type Customer = {
  customer_id: string;
  name: string;
  segment: string;
  brand: string;
  lifetime_value: number;
  match_confidence: number;
  preferred_channel: string;
  interests: Interest[];
  intent_score: number;
  fatigue_score: number;
  diagnosis: string;
  fatigue_cause: string;
  favorite_driver?: string;
  favorite_brand?: string;
};

type EventRow = {
  type: string;
  channel: string;
  product_name: string;
  sentiment: string;
  created_at: string;
};

type ModelStatus = {
  intent_model: { version: number; last_trained: string | null; samples: number };
  diagnosis_model: { version: number; last_trained: string | null; samples: number };
  nba_model: { version: number; last_trained: string | null; samples: number };
  retrain_count: number;
  last_retrain: string | null;
};

type Dashboard = {
  store: string;
  metrics: Record<string, number>;
  channel_performance: { channel: string; ctr: number; fatigue: number }[];
  recommendations: Recommendation[];
  model_status: ModelStatus;
  metrics_history: Record<string, number | string>[];
};

type Detail = {
  customer: Customer;
  events: EventRow[];
  recommendation: Recommendation;
  action_history: { action_taken: string; label: string; created_at: string }[];
};

function App() {
  const [tab, setTab] = useState("dashboard");
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  async function load() {
    setError("");
    try {
      const [dashboardRes, customersRes] = await Promise.all([
        fetch(`${API}/api/dashboard`),
        fetch(`${API}/api/customers`),
      ]);
      if (!dashboardRes.ok || !customersRes.ok) throw new Error("API not responding");
      const dashboardData = await dashboardRes.json();
      const customersData = await customersRes.json();
      setDashboard(dashboardData);
      setCustomers(customersData);
      const targetId = selectedCustomer || customersData[0]?.customer_id;
      if (targetId) {
        const detailRes = await fetch(`${API}/api/customers/${targetId}`);
        if (detailRes.ok) setDetail(await detailRes.json());
      }
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load data");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!selectedCustomer) return;
    fetch(`${API}/api/customers/${selectedCustomer}`)
      .then((res) => res.json())
      .then(setDetail)
      .catch(() => null);
  }, [selectedCustomer]);

  const pending = useMemo(
    () => dashboard?.recommendations.filter((rec) => rec.status === "pending") ?? [],
    [dashboard],
  );

  return (
    <div className="min-h-screen">
      <header className="border-b border-border bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Activity size={22} />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Pulse</h1>
              <p className="text-sm text-muted-foreground">
                LightGBM intent · Fatigue diagnosis · NBA decision tree · FAISS explainability
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge className="bg-teal-50 text-teal-800">
              <Database size={14} /> {dashboard?.store ?? "Loading..."}
            </Badge>
            {lastUpdated ? (
              <Badge className="bg-slate-100 text-slate-700">
                <Clock size={14} /> Updated {lastUpdated}
              </Badge>
            ) : null}
            <GhostButton onClick={load} disabled={loading}>
              <RefreshCw size={16} /> Refresh
            </GhostButton>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-5">
        <nav className="mb-5 flex flex-wrap gap-2">
          {[
            ["dashboard", "Dashboard", Gauge],
            ["queue", "Review Queue", BadgeCheck],
            ["customers", "Customers", UserRound],
            ["simulator", "Simulator", SlidersHorizontal],
            ["models", "Models", Layers],
          ].map(([id, label, Icon]) => (
            <button
              key={String(id)}
              onClick={() => setTab(String(id))}
              className={`inline-flex h-10 items-center gap-2 rounded-md border px-3 text-sm font-semibold ${
                tab === id ? "border-primary bg-primary text-primary-foreground" : "border-border bg-white"
              }`}
            >
              <Icon size={16} /> {String(label)}
            </button>
          ))}
        </nav>

        {error ? (
          <Card className="mb-4 border-destructive p-4 text-sm text-destructive">
            {error}. Start backend:{" "}
            <code>cd backend && python -m uvicorn app.main:app --port 8003</code>
          </Card>
        ) : null}
        {loading && !dashboard ? (
          <Card className="p-6 text-sm text-muted-foreground">Initializing pipeline...</Card>
        ) : null}

        {dashboard && tab === "dashboard" ? (
          <DashboardView dashboard={dashboard} pending={pending.length} />
        ) : null}
        {dashboard && tab === "queue" ? (
          <QueueView recommendations={dashboard.recommendations} onChanged={load} />
        ) : null}
        {tab === "customers" ? (
          <CustomerView
            customers={customers}
            selected={selectedCustomer}
            setSelected={setSelectedCustomer}
            detail={detail}
          />
        ) : null}
        {tab === "simulator" ? (
          <SimulatorView
            customers={customers}
            selected={selectedCustomer}
            setSelected={setSelectedCustomer}
            onChanged={load}
          />
        ) : null}
        {dashboard && tab === "models" ? <ModelsView status={dashboard.model_status} onRetrain={load} /> : null}
      </main>
    </div>
  );
}

function DashboardView({ dashboard, pending }: { dashboard: Dashboard; pending: number }) {
  const metrics = [
    ["Events/day", dashboard.metrics.events_per_day, "Daily behavioral signals"],
    ["ID match", `${dashboard.metrics.id_match_rate}%`, "PeopleCloud identity confidence"],
    ["Suppression", `${dashboard.metrics.suppression_rate}%`, "Users in cooldown"],
    ["Interaction", `${dashboard.metrics.interaction_rate}%`, "Positive outcome rate"],
    ["Override", `${dashboard.metrics.override_rate}%`, "Marketer HITL feedback"],
    ["Retrains", dashboard.model_status.retrain_count, "Mini-batch cycles completed"],
  ];

  const historyChart = dashboard.metrics_history
    .filter((h) => h.interaction_rate !== undefined)
    .slice(-10)
    .map((h, i) => ({
      tick: i + 1,
      interaction: Number(h.interaction_rate),
      suppression: Number(h.suppression_rate),
    }));

  return (
    <div className="space-y-5">
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
        {metrics.map(([label, value, helper]) => (
          <Card key={label} className="p-4">
            <p className="text-xs font-semibold uppercase text-muted-foreground">{label}</p>
            <p className="mt-2 text-2xl font-bold">{value}</p>
            <p className="mt-1 text-xs text-muted-foreground">{helper}</p>
          </Card>
        ))}
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
        <Card className="p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-bold">Channel Health</h2>
            <Badge>{pending} pending reviews</Badge>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dashboard.channel_performance}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="channel" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="ctr" fill="#0f9f8e" name="CTR %" radius={[4, 4, 0, 0]} />
                <Bar dataKey="fatigue" fill="#f59e0b" name="Fatigue %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-5">
          <h2 className="mb-4 text-lg font-bold">Live Metrics Trend</h2>
          <div className="h-72">
            {historyChart.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyChart}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="tick" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="interaction" stroke="#0f9f8e" name="Interaction %" />
                  <Line type="monotone" dataKey="suppression" stroke="#f59e0b" name="Suppression %" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted-foreground">Metrics will populate as retrain cycles run.</p>
            )}
          </div>
        </Card>
      </section>
    </div>
  );
}

function Insight({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) {
  return (
    <Card className="flex gap-3 p-4">
      <div className="mt-0.5 text-primary">{icon}</div>
      <div>
        <p className="font-semibold">{title}</p>
        <p className="text-sm text-muted-foreground">{text}</p>
      </div>
    </Card>
  );
}

function QueueView({ recommendations, onChanged }: { recommendations: Recommendation[]; onChanged: () => void }) {
  const pending = recommendations.filter((r) => r.status === "pending");
  return (
    <div>
      <p className="mb-4 text-sm text-muted-foreground">{pending.length} recommendations awaiting marketer review</p>
      <div className="grid gap-4 lg:grid-cols-3">
        {pending.slice(0, 12).map((rec) => (
          <RecommendationCard key={rec.id} rec={rec} onChanged={onChanged} />
        ))}
      </div>
    </div>
  );
}

function RecommendationCard({ rec, onChanged }: { rec: Recommendation; onChanged: () => void }) {
  const [reason, setReason] = useState("");
  const [editing, setEditing] = useState(false);
  const [action, setAction] = useState(rec.action);
  const [channel, setChannel] = useState(rec.channel);
  const [timing, setTiming] = useState(rec.timing);

  async function submit(status: string) {
    await fetch(`${API}/api/recommendations/${rec.id}/override`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, reason, edited_action: action, edited_channel: channel, edited_timing: timing }),
    });
    onChanged();
  }

  return (
    <Card className="flex flex-col p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <p className="text-sm text-muted-foreground">{rec.brand}</p>
          <h3 className="text-lg font-bold">{rec.customer_name}</h3>
        </div>
        <div className="flex flex-col items-end gap-2">
          <Badge className="bg-indigo-50 text-indigo-800">{rec.nba_action}</Badge>
          <Badge className={rec.suppression ? "bg-amber-50 text-amber-800" : "bg-teal-50 text-teal-800"}>
            {rec.status}
          </Badge>
        </div>
      </div>
      <div className="space-y-2 text-sm">
        <p className="font-semibold">{rec.action}</p>
        <p className="text-muted-foreground">{rec.explanation}</p>
        <p className="text-xs text-muted-foreground">{rec.diagnosis}</p>
        <Badge className="bg-slate-100 text-slate-700">Cause: {rec.fatigue_cause}</Badge>
      </div>
      {rec.top_products?.length ? (
        <div className="my-3 rounded-md bg-muted p-2 text-xs">
          <p className="font-semibold">Top-k products (≥0.85 combined)</p>
          {rec.top_products.map((p) => (
            <p key={p.product_id} className="text-muted-foreground">
              {p.name} — score {p.combined_score}
            </p>
          ))}
        </div>
      ) : null}
      <div className="my-4 grid grid-cols-3 gap-2 text-center text-sm">
        <Score label="Intent" value={rec.intent_score} />
        <Score label="Fatigue" value={rec.fatigue_score} />
        <Score label="Channel" value={rec.channel} />
      </div>
      {editing ? (
        <div className="mb-3 space-y-2">
          <Input value={action} onChange={(e) => setAction(e.target.value)} />
          <Select value={channel} onChange={(e) => setChannel(e.target.value)}>
            {["email", "push", "paid", "onsite"].map((item) => (
              <option key={item}>{item}</option>
            ))}
          </Select>
          <Input value={timing} onChange={(e) => setTiming(e.target.value)} />
        </div>
      ) : null}
      <Select
        className="mb-2"
        value={reason}
        onChange={(e) => setReason(e.target.value)}
      >
        <option value="">Rejection reason (if rejecting)</option>
        <option value="wrong_audience">Wrong audience</option>
        <option value="wrong_timing">Wrong timing</option>
        <option value="wrong_offer">Wrong offer</option>
        <option value="brand_guideline">Brand guideline conflict</option>
      </Select>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button onClick={() => submit("approved")}>
          <Check size={16} /> Approve
        </Button>
        <GhostButton onClick={() => submit("rejected")}>
          <X size={16} /> Reject
        </GhostButton>
        <GhostButton onClick={() => setEditing((v) => !v)}>
          <SlidersHorizontal size={16} /> Edit
        </GhostButton>
      </div>
    </Card>
  );
}

function Score({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md bg-muted p-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 font-bold">{value}</p>
    </div>
  );
}

function CustomerView({
  customers,
  selected,
  setSelected,
  detail,
}: {
  customers: Customer[];
  selected: string;
  setSelected: (id: string) => void;
  detail: Detail | null;
}) {
  return (
    <div className="grid gap-5 lg:grid-cols-[320px_1fr]">
      <Card className="max-h-[70vh] overflow-y-auto p-3">
        <div className="space-y-2">
          {customers.map((customer) => (
            <button
              key={customer.customer_id}
              onClick={() => setSelected(customer.customer_id)}
              className={`w-full rounded-md border p-3 text-left ${
                selected === customer.customer_id ? "border-primary bg-teal-50" : "border-border bg-white"
              }`}
            >
              <p className="font-semibold">{customer.name}</p>
              <p className="text-sm text-muted-foreground">
                {customer.segment} · {customer.brand}
              </p>
              <p className="text-xs text-muted-foreground">Fatigue: {customer.fatigue_cause}</p>
            </button>
          ))}
        </div>
      </Card>
      {detail ? (
        <div className="space-y-5">
          <Card className="p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{detail.customer.brand}</p>
                <h2 className="text-2xl font-bold">{detail.customer.name}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {detail.customer.interests.map((i) => `${i.entity} (${i.category})`).join(", ")}
                </p>
                {detail.customer.favorite_driver ? (
                  <p className="text-xs text-muted-foreground">Driver: {detail.customer.favorite_driver}</p>
                ) : null}
              </div>
              <Badge>${detail.customer.lifetime_value} LTV</Badge>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-4">
              <Score label="Intent" value={detail.recommendation.intent_score} />
              <Score label="Fatigue" value={detail.recommendation.fatigue_score} />
              <Score label="ID match" value={`${Math.round(detail.customer.match_confidence * 100)}%`} />
              <Score label="Channel" value={detail.recommendation.channel} />
            </div>
          </Card>

          <Card className="p-5">
            <h3 className="mb-3 text-lg font-bold">NBA Recommendation</h3>
            <p className="font-semibold">{detail.recommendation.action}</p>
            <p className="mt-2 text-sm text-muted-foreground">{detail.recommendation.explanation}</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <Badge>{detail.recommendation.nba_action}</Badge>
              <Badge>{detail.recommendation.fatigue_cause}</Badge>
              <Badge>v{detail.recommendation.model_version}</Badge>
            </div>
          </Card>

          <Card className="p-5">
            <h3 className="mb-3 text-lg font-bold">Signal Timeline</h3>
            <div className="space-y-3">
              {detail.events.map((event, index) => (
                <div
                  key={`${event.created_at}-${index}`}
                  className="flex flex-col gap-2 rounded-md border border-border p-3 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div>
                    <p className="font-semibold">{event.type.replaceAll("_", " ")}</p>
                    <p className="text-sm text-muted-foreground">
                      {event.product_name} via {event.channel}
                    </p>
                  </div>
                  <Badge
                    className={
                      event.sentiment === "positive" ? "bg-teal-50 text-teal-800" : "bg-amber-50 text-amber-800"
                    }
                  >
                    {event.sentiment}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>
        </div>
      ) : null}
    </div>
  );
}

function SimulatorView({
  customers,
  selected,
  setSelected,
  onChanged,
}: {
  customers: Customer[];
  selected: string;
  setSelected: (id: string) => void;
  onChanged: () => void;
}) {
  const [type, setType] = useState("email_ignore");
  const [channel, setChannel] = useState("email");
  const [product, setProduct] = useState("Ferrari Team Cap");
  const [sentiment, setSentiment] = useState("negative");
  const [result, setResult] = useState<Recommendation | null>(null);

  async function run() {
    const res = await fetch(`${API}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        customer_id: selected,
        type,
        channel,
        product,
        sentiment,
        offer_category: "sports",
      }),
    });
    const data = await res.json();
    setResult(data.recommendation);
    onChanged();
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[420px_1fr]">
      <Card className="p-5">
        <h2 className="text-lg font-bold">Outcome Simulation</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Inject a behavioral signal, capture outcome label, and watch models update the NBA.
        </p>
        <div className="mt-4 space-y-3">
          <Select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {customers.map((c) => (
              <option key={c.customer_id} value={c.customer_id}>
                {c.name}
              </option>
            ))}
          </Select>
          <Select value={type} onChange={(e) => setType(e.target.value)}>
            {["email_ignore", "ad_skip", "unsubscribe", "search", "cart_add", "click", "purchase", "view"].map(
              (item) => (
                <option key={item}>{item}</option>
              ),
            )}
          </Select>
          <Select value={channel} onChange={(e) => setChannel(e.target.value)}>
            {["email", "push", "paid", "onsite"].map((item) => (
              <option key={item}>{item}</option>
            ))}
          </Select>
          <Input value={product} onChange={(e) => setProduct(e.target.value)} />
          <Select value={sentiment} onChange={(e) => setSentiment(e.target.value)}>
            <option>negative</option>
            <option>positive</option>
            <option>neutral</option>
          </Select>
          <Button className="w-full" onClick={run}>
            <Send size={16} /> Run Simulation
          </Button>
        </div>
      </Card>
      <Card className="p-5">
        <h2 className="mb-3 text-lg font-bold">Updated Decision</h2>
        {result ? (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Badge>
                <Clock size={14} /> {result.timing}
              </Badge>
              <Badge>{result.channel}</Badge>
              <Badge>{result.nba_action}</Badge>
              <Badge>{result.fatigue_cause}</Badge>
            </div>
            <h3 className="text-2xl font-bold">{result.action}</h3>
            <p className="text-muted-foreground">{result.explanation}</p>
            <p className="text-sm text-muted-foreground">{result.diagnosis}</p>
            <div className="grid gap-3 sm:grid-cols-2">
              <Score label="Intent" value={result.intent_score} />
              <Score label="Fatigue" value={result.fatigue_score} />
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No simulation run yet.</p>
        )}
      </Card>
    </div>
  );
}

function ModelsView({ status, onRetrain }: { status: ModelStatus; onRetrain: () => void }) {
  const [retraining, setRetraining] = useState(false);

  async function triggerRetrain() {
    setRetraining(true);
    await fetch(`${API}/api/retrain`, { method: "POST" });
    onRetrain();
    setRetraining(false);
  }

  const models = [
    { name: "Intent Scoring", key: "intent_model", type: "LightGBM", desc: "P(conversion) per user × product" },
    { name: "Fatigue Diagnosis", key: "diagnosis_model", type: "RandomForest", desc: "Channel vs timing vs offer fatigue" },
    { name: "NBA Policy", key: "nba_model", type: "DecisionTree", desc: "Action selection with combination fallback" },
  ];

  return (
    <div className="space-y-5">
      <Card className="flex items-center justify-between p-5">
        <div>
          <h2 className="text-lg font-bold">Model Registry</h2>
          <p className="text-sm text-muted-foreground">
            Mini-batch retrain every 60 min · Last: {status.last_retrain ?? "pending"}
          </p>
        </div>
        <Button onClick={triggerRetrain} disabled={retraining}>
          <Zap size={16} /> {retraining ? "Retraining..." : "Trigger Retrain Now"}
        </Button>
      </Card>
      <div className="grid gap-4 lg:grid-cols-3">
        {models.map((m) => {
          const info = status[m.key as keyof ModelStatus] as { version: number; samples: number; last_trained: string | null };
          return (
            <Card key={m.key} className="p-5">
              <Badge className="bg-indigo-50 text-indigo-800">{m.type}</Badge>
              <h3 className="mt-2 text-lg font-bold">{m.name}</h3>
              <p className="text-sm text-muted-foreground">{m.desc}</p>
              <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                <Score label="Version" value={info.version} />
                <Score label="Samples" value={info.samples} />
              </div>
            </Card>
          );
        })}
      </div>
      <Card className="p-5">
        <h3 className="font-bold">Retraining Pipeline</h3>
        <pre className="mt-3 overflow-x-auto rounded-md bg-muted p-4 text-xs">
{`Events stream in
        ↓
Collect last 60 min
        ↓
Retrain LightGBM + Diagnosis + NBA
        ↓
Refresh FAISS index + recommendations`}
        </pre>
        <p className="mt-2 text-sm text-muted-foreground">
          Total retrain cycles: {status.retrain_count}
        </p>
      </Card>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
