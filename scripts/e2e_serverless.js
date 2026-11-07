const { chromium } = require('playwright');
const { execSync } = require('child_process');
const OUT = process.env.E2E_OUT || '.';
const KILL_A = process.env.E2E_KILL_A;
const TOKEN = 'e2eSessionToken_' + 'x'.repeat(24);
const Q = 'Why did our revenue decline last quarter, and which customer segments contributed most?';
(async () => {
  const b = await chromium.launch(); const out = {}; const issues = [];
  const page = async (port) => {
    const ctx = await b.newContext({ viewport: { width: 1440, height: 1000 } });
    await ctx.addInitScript((t) => localStorage.setItem('databeacon.session', t), TOKEN);
    const p = await ctx.newPage();
    p.on('console', m => { if (m.type() === 'error' || /Refused to|Content Security/.test(m.text())) issues.push(port + ': ' + m.text()); });
    p.on('pageerror', e => issues.push(port + ': ' + String(e)));
    await p.goto('http://localhost:' + port + '/'); await p.locator('.brand').waitFor({ timeout: 30000 }); await p.waitForTimeout(1500);
    return p;
  };
  // 1. replay
  console.log('STEP 1. replay');
  const a = await page(8000);
  await a.getByText('Or watch a recorded investigation').waitFor();
  await a.locator('button.card', { hasText: 'Why did our revenue decline' }).first().click();
  await a.getByText('Replaying the agent').waitFor();
  await a.screenshot({ path: '' + OUT + 'replay_playing.png' });
  await a.getByRole('button', { name: 'Show everything' }).click();
  await a.getByText('Answer', { exact: true }).waitFor();
  await a.waitForTimeout(1200);
  out.replayCharts = await a.locator('.js-plotly-plot .main-svg').count();
  out.replayAnswerHasEnterprise = (await a.locator('.summary').innerText()).includes('Enterprise');
  out.replayPdf = await a.evaluate(async () => { const l = [...document.querySelectorAll('a')].find(x => x.textContent.includes('Download PDF')); const r = await fetch(l.href); return r.status + ' ' + r.headers.get('content-type'); });
  await a.screenshot({ path: '' + OUT + 'replay_done.png', fullPage: true });
  // 2. live inline run to completion
  console.log('STEP 2. live inline run to completion');
  await a.getByRole('button', { name: 'Analyze' }).click();
  await a.getByRole('button', { name: 'Try it with sample e-commerce data' }).click();
  await a.locator('textarea').first().fill(Q);
  await a.getByRole('button', { name: 'Run analysis' }).click();
  await a.getByText('Download PDF report').waitFor({ timeout: 90000 });
  out.liveCompleted = await a.locator('.badge', { hasText: 'completed' }).count() > 0;
  // 3. reconnect
  console.log('STEP 3. reconnect'); //: start on A, open on B mid-run
  await a.getByRole('button', { name: 'New' }).click();
  await a.locator('textarea').first().fill(Q);
  await a.getByRole('button', { name: 'Run analysis' }).click();
  await a.waitForTimeout(5000);
  const bpage = await page(8001);
  await bpage.locator('a', { hasText: 'Why did our revenue decline' }).first().click();
  await bpage.getByText('Reconnected to saved progress').waitFor({ timeout: 15000 });
  out.reconnectBanner = true;
  await bpage.screenshot({ path: '' + OUT + 'reconnected.png' });
  await bpage.getByText('Download PDF report').waitFor({ timeout: 90000 });
  out.reconnectedRunCompleted = await bpage.locator('.badge', { hasText: 'completed' }).count() > 0;
  // 4. interrupted
  console.log('STEP 4. interrupted'); //: kill the executor mid-run, look from B
  await a.getByRole('button', { name: 'New' }).click();
  await a.locator('textarea').first().fill(Q);
  await a.getByRole('button', { name: 'Run analysis' }).click();
  await a.waitForTimeout(4000);
  execSync(KILL_A);
  await bpage.reload();
  await bpage.waitForTimeout(8000);  // heartbeat goes stale (RUN_STALE_S=6)
  await bpage.reload(); await bpage.waitForTimeout(2500);
  const runs = bpage.locator('a', { hasText: 'Why did our revenue decline' });
  await runs.first().click();
  await bpage.getByText('This investigation was interrupted').waitFor({ timeout: 20000 });
  out.interruptedBanner = true;
  out.interruptionReason = await bpage.locator('[role=alert]').innerText();
  await bpage.screenshot({ path: '' + OUT + 'interrupted.png' });
  await bpage.getByRole('button', { name: 'Start it again' }).click();
  await bpage.getByText('Download PDF report').waitFor({ timeout: 90000 });
  out.rerunCompleted = await bpage.locator('.badge', { hasText: 'completed' }).count() > 0;
  out.issues = issues;
  console.log(JSON.stringify(out, null, 1));
  await b.close();
})().catch(e => { console.error('E2E FAILED:', e.message); process.exit(1); });
