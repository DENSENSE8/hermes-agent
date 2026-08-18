#!/usr/bin/env node
/**
 * Paperclip Adapter Entry — Unified Memory Agent Bootstrap
 *
 * Loads the hermes-paperclip-adapter, configures it with the local
 * memory infrastructure (Qdrant + Honcho + Memory Sidecar), and
 * registers Hermes as the central memory agent for all sources:
 * nemoclaw, openclaw, paperclip, sidecar.
 *
 * Usage:
 *   node paperclip-adapter-entry.mjs
 *   node paperclip-adapter-entry.mjs --config /path/to/config.json
 */

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Load Config ─────────────────────────────────────────────────────────────

const configPath = process.argv.includes('--config')
  ? process.argv[process.argv.indexOf('--config') + 1]
  : process.env.PAPERCLIP_ADAPTER_CONFIG
    || resolve(__dirname, 'paperclip-adapter.config.json');

if (!existsSync(configPath)) {
  console.error(`[paperclip-adapter] Config not found: ${configPath}`);
  process.exit(1);
}

const config = JSON.parse(readFileSync(configPath, 'utf-8'));
console.log(`[paperclip-adapter] Loaded config: ${config.name}`);
console.log(`[paperclip-adapter] Memory mode: ${config.adapterConfig.environment.PAPERCLIP_MEMORY_MODE}`);
console.log(`[paperclip-adapter] Sources: ${config.adapterConfig.environment.PAPERCLIP_SOURCES}`);

// ── Inject Environment ──────────────────────────────────────────────────────

for (const [key, val] of Object.entries(config.adapterConfig.environment)) {
  if (!process.env[key]) {
    process.env[key] = val;
  }
}

// ── Verify Memory Sidecar ───────────────────────────────────────────────────

async function verifySidecar() {
  const sidecarUrl = process.env.MEMORY_SIDECAR_URL || 'http://localhost:8791';
  try {
    const res = await fetch(`${sidecarUrl}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    console.log(`[paperclip-adapter] Sidecar health:`, data.status);
    console.log(`[paperclip-adapter] Registered sources:`, data.registered_sources?.join(', ') || 'unknown');

    if (data.status !== 'healthy') {
      console.warn(`[paperclip-adapter] WARNING: Sidecar is ${data.status}`);
      for (const [check, status] of Object.entries(data.checks || {})) {
        if (status !== 'ok') console.warn(`  - ${check}: ${status}`);
      }
    }
    return data;
  } catch (err) {
    console.error(`[paperclip-adapter] Sidecar unreachable at ${sidecarUrl}: ${err.message}`);
    return null;
  }
}

// ── Register Paperclip Source ───────────────────────────────────────────────

async function registerPaperclipSource() {
  const sidecarUrl = process.env.MEMORY_SIDECAR_URL || 'http://localhost:8791';
  try {
    // Save a registration marker
    const res = await fetch(`${sidecarUrl}/api/memory/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: 'paperclip',
        query: 'Paperclip adapter registered as unified memory agent',
        response: `Hermes is now the central memory agent for: ${config.adapterConfig.environment.PAPERCLIP_SOURCES}. Config: ${configPath}`,
        model: config.adapterConfig.model,
        metadata: {
          type: 'adapter-registration',
          adapter: 'hermes-paperclip-adapter',
          sources: config.adapterConfig.environment.PAPERCLIP_SOURCES.split(','),
          timestamp: new Date().toISOString(),
        },
      }),
      signal: AbortSignal.timeout(10000),
    });
    const data = await res.json();
    console.log(`[paperclip-adapter] Registration saved:`, data.id);
    return data;
  } catch (err) {
    console.warn(`[paperclip-adapter] Registration save failed: ${err.message}`);
    return null;
  }
}

// ── Load Adapter ────────────────────────────────────────────────────────────

async function loadAdapter() {
  try {
    const adapter = await import('hermes-paperclip-adapter');
    console.log(`[paperclip-adapter] Adapter module loaded`);
    return adapter;
  } catch (err) {
    console.error(`[paperclip-adapter] Failed to import adapter: ${err.message}`);
    return null;
  }
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log('[paperclip-adapter] Starting unified memory agent...');
  console.log(`[paperclip-adapter] Hermes home: ${__dirname}`);
  console.log(`[paperclip-adapter] Config: ${configPath}`);

  // 1. Verify memory sidecar is up
  const health = await verifySidecar();

  // 2. Load the paperclip adapter module
  const adapter = await loadAdapter();

  // 3. Register paperclip as a memory source
  await registerPaperclipSource();

  // 4. Report status
  console.log('\n[paperclip-adapter] === Unified Memory Agent Status ===');
  console.log(`  Agent:     ${config.name}`);
  console.log(`  Model:     ${config.adapterConfig.model}`);
  console.log(`  Provider:  ${config.adapterConfig.provider}`);
  console.log(`  Sidecar:   ${health?.status || 'unreachable'}`);
  console.log(`  Qdrant:    ${health?.checks?.qdrant || 'unknown'}`);
  console.log(`  Honcho:    ${health?.checks?.honcho || 'unknown'}`);
  console.log(`  Ollama:    ${health?.checks?.ollama || 'unknown'}`);
  console.log(`  Sources:   ${Object.keys(config.memorySources).join(', ')}`);
  console.log(`  Adapter:   ${adapter ? 'loaded' : 'not loaded (standalone mode)'}`);
  console.log('[paperclip-adapter] Ready.\n');

  return { config, health, adapter };
}

main().catch(err => {
  console.error('[paperclip-adapter] Fatal:', err);
  process.exit(1);
});
