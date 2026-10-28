// One FinDoc reviewer per VU: log in once, create a chat, upload a PDF, poll
// until ingestion finishes, then keep browsing every 5–10 s. Driven by
// load_test_k6.py, which owns the accounts, server, stub workers and cleanup.
import http from 'k6/http';
import { sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE = __ENV.BASE;
const PREFIX = __ENV.PREFIX;
const PASSWORD = __ENV.PASSWORD;
const PDF = open(__ENV.PDF, 'b');
const ENDPOINTS = ['login', 'health', 'chats', 'chat', 'create_chat', 'upload', 'job_status'];

const latency = {};
const failed = {};
for (const name of ENDPOINTS) {
  latency[name] = new Trend(`api_${name}`, true);
  failed[name] = new Rate(`api_${name}_failed`);
}
const jobTotal = new Trend('job_total', true);

export const options = {
  stages: JSON.parse(__ENV.STAGES),
  summaryTrendStats: ['p(50)', 'p(95)', 'p(99)', 'max', 'count'],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    api_health: ['p(95)<2000'],
    api_chats: ['p(95)<5000'],
    api_chat: ['p(95)<5000'],
    api_job_status: ['p(95)<5000'],
    api_upload: ['p(95)<10000'],
  },
};

function call(name, method, path, body, params) {
  const res = http.request(method, `${BASE}${path}`, body, params);
  latency[name].add(res.timings.duration);
  failed[name].add(res.status === 0 || res.status >= 400);
  return res;
}

let session = null;

function start() {
  const login = call('login', 'POST', '/api/auth/login',
    JSON.stringify({ username: `${PREFIX}${__VU}`, password: PASSWORD }),
    { headers: { 'Content-Type': 'application/json' } });
  if (login.status !== 200) return null;
  const auth = { headers: { Authorization: `Bearer ${login.json('token')}` } };

  const created = call('create_chat', 'POST', '/api/chats/new', null, auth);
  if (created.status !== 200) return null;
  const chatId = created.json('chat.id');

  const upload = call('upload', 'POST', `/api/chats/${chatId}/document`,
    { file: http.file(PDF, 'load-test.pdf', 'application/pdf') },
    { headers: { ...auth.headers, 'Idempotency-Key': `${PREFIX}${__VU}-${Date.now()}` } });
  return { auth, chatId, uploadedAt: Date.now(), ingesting: upload.status === 202 };
}

export default function () {
  if (!session) {
    session = start();
    if (!session) { sleep(5); return; }
  }
  const { auth, chatId } = session;

  if (session.ingesting) {
    const res = call('job_status', 'GET', `/api/chats/${chatId}/status`, null, auth);
    const status = res.status === 200 ? res.json('status') : null;
    if (status === 'ready' || status === 'failed') {
      if (status === 'ready') jobTotal.add(Date.now() - session.uploadedAt);
      session.ingesting = false;
    }
    sleep(2);
    return;
  }

  call('health', 'GET', '/api/health');
  call('chats', 'GET', '/api/chats', null, auth);
  call('chat', 'GET', `/api/chats/${chatId}`, null, auth);
  sleep(5 + Math.random() * 5);
}

export function handleSummary(data) {
  return { [__ENV.OUT]: JSON.stringify(data) };
}
