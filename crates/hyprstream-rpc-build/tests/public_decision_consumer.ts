// Executed ONLY over outputs freshly generated from canonical CGRs by the
// same reviewed codegen binary in public_ts_consumer.rs. No handwritten client
// or wire encoding substitutes for these actual generated builders/parsers.
import { ModelClient, RegistryClient, RpcError } from './index';
import { DecisionRequest, DecisionResponse, RpcTransport } from './model';
import {
  DecisionBatch, Entry, NoulCriteria, QuestionSet, buildQuestionSet, parseQuestionSet, buildDecisionBatch, parseDecisionBatch,
  buildNoulCriteria, parseNoulCriteria, buildOptEntry_none, buildOptEntry_some, parseOptEntry,
} from './decision';
import {
  buildModelResponse_error, buildModelResponse_inferResult, parseModelRequest,
} from './model_wire_fixture';
import {
  CtlRequest, buildRegistryResponse_repoResult, parseRegistryRequest,
  buildGetBlobRequest_xetMerkle, parseGetBlobRequest,
  buildWorktreeRequest_ctl, parseWorktreeRequest,
} from './registry_wire_fixture';
import {
  GroupNode, EnumNode, LeafParent, NestedLeaf, NamedGroupNode, buildNamedGroupNode, parseNamedGroupNode,
  buildGroupNode_branch, parseGroupNode,
  buildEnumNode_next, buildEnumNode_color, parseEnumNode,
  buildLeafParent_next, buildLeafParent_leaf, parseLeafParent,
} from './recursive_codegen_fixture';

function ensure(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const request: DecisionRequest = {
  clefJson: '{"model":"clef-flash-fp8:main","state":{"n":1000000000000000100},"questions":{"z":{"type":"noul"},"a":{"type":"noul"}}}',
  maxStateTokens: 0,
  timeoutMs: 17,
};
const batch: DecisionBatch = {
  version: { schema: 'wire-fixture-schema', model: 'wire-fixture-model', calib: undefined },
  rows: [{ answers: [
    { questionId: 'z', value: { variant: 'noul', data: Math.fround(0.1) }, conformalSet: undefined },
    { questionId: 'a', value: { variant: 'choice', data: [Math.fround(0.1), Math.fround(0.9)] }, conformalSet: [] },
  ] }],
};
const decision: DecisionResponse = {
  batch,
  provenance: {
    encodingVersion: 'wire-fixture-encoding', modelRef: 'clef-flash-fp8:main', artifactCommit: 'wire-fixture-commit',
    headWeightsSha256: 'wire-fixture-head', headConfigSha256: 'wire-fixture-head-config',
    tokenizerSha256: 'wire-fixture-tokenizer', backboneConfigSha256: 'wire-fixture-config',
    quantization: 'wire-fixture-quantization', headTensorCount: 122, computeProfile: 'wire-fixture-cpu',
  },
  limits: { maxLength: 1024, maxStateTokens: 0, stateTokensAdmitted: 0, sequenceTokens: 31, stateTruncated: true },
};
let response = buildModelResponse_error(1n, { message: 'outer input rejected', code: 'DECISION_INPUT', details: '' });
let transportFailure: Error | undefined;
const streamingFixtureFailure = new Error('wire fixture stream stopped before delivery');
const calls: { service: string; method: number; bytes: Uint8Array; options: boolean }[] = [];
const transport: RpcTransport = {
  nextId: () => 1n,
  async callForServiceWithMethod(service, method, bytes) {
    calls.push({ service, method, bytes, options: false });
    if (transportFailure) throw transportFailure;
    return response;
  },
  async callWithOptionsForServiceWithMethod(service, method, bytes) {
    calls.push({ service, method, bytes, options: true });
    if (transportFailure) throw transportFailure;
    return response;
  },
  async openStreamForServiceWithMethod(service, method, bytes) {
    calls.push({ service, method, bytes, options: false });
    throw streamingFixtureFailure;
  },
  async openStreamWithOptionsForServiceWithMethod(service, method, bytes) {
    calls.push({ service, method, bytes, options: true });
    throw streamingFixtureFailure;
  },
};
const client = new ModelClient(transport);

async function remoteError(action: () => Promise<unknown>, code: string, message: string) {
  try { await action(); throw new Error('missing actual generated remote exception'); }
  catch (error) {
    ensure(error instanceof RpcError, 'decoded error must use the shared runtime class');
    ensure(error instanceof Error, 'remote error remains Error compatible');
    ensure(error.name === 'Error', 'default Error name must remain unchanged');
    ensure(error.code === code && error.message === message, 'wire code/message must survive unchanged');
  }
}

async function main() {
  let named: NamedGroupNode = { label: 'end-label', body: { variant: 'end', data: undefined } };
  for (let depth = 0; depth < 8; depth++) named = { label: `node-${depth}`, body: { variant: 'next', data: named } };
  ensure(JSON.stringify(parseNamedGroupNode(buildNamedGroupNode(named))) === JSON.stringify(named), 'named-group-only cycle must preserve enclosing fields and inline union');
  const nullEntry: Entry = { variant: 'null', data: undefined };
  const questions: QuestionSet = { state: { variant: 'none', data: undefined }, questions: [
    { id: 'choice-z', kind: 'choice', instructions: { variant: 'some', data: nullEntry },
      body: { variant: 'choice', data: [{ name: 'z', rubric: { variant: 'some', data: nullEntry } }, { name: 'a', rubric: { variant: 'none', data: undefined } }] } },
    { id: 'score-a', kind: 'score', instructions: { variant: 'none', data: undefined },
      body: { variant: 'score', data: [{ variant: 'none', data: undefined }, { variant: 'some', data: { variant: 'number', data: 1.5 } }, { variant: 'some', data: nullEntry }] } },
  ] };
  ensure(JSON.stringify(parseQuestionSet(buildQuestionSet(questions))) === JSON.stringify(questions), 'actual QuestionSpec named body union must share original parent offsets');
  const ctl: CtlRequest = { fid: 17, variant: 'log', data: { maxCount: 7, refName: 'main' } };
  ensure(JSON.stringify(parseWorktreeRequest(buildWorktreeRequest_ctl('main', ctl))) === JSON.stringify({ name: 'main', variant: 'ctl', data: ctl }), 'actual Registry mixed scoped request retains shared fields and nested union');
  ensure(JSON.stringify(parseGetBlobRequest(buildGetBlobRequest_xetMerkle('wire-fixture-repo', 'wire-fixture-cid'))) === JSON.stringify({ grantRepo: 'wire-fixture-repo', variant: 'xetMerkle', data: 'wire-fixture-cid' }), 'actual GetBlob mixed value union remains typed and complete');

  // Independent canonical schemas cover group-only cycle edges, nonzero enum
  // union data and noncyclic union payloads under a recursive parent.
  let group: GroupNode = { variant: 'end', data: undefined };
  for (let depth = 0; depth < 8; depth++) group = { variant: 'branch', data: { next: group } };
  const groupDecoded = parseGroupNode(buildGroupNode_branch({ next: group }));
  ensure(JSON.stringify(groupDecoded) === JSON.stringify({ variant: 'branch', data: { next: group } }), 'group-only recursion must preserve every node');
  const colored: EnumNode = { variant: 'next', data: { variant: 'color', data: 'green' } };
  ensure(JSON.stringify(parseEnumNode(buildEnumNode_next(colored))) === JSON.stringify({ variant: 'next', data: colored }), 'nonzero enum union data inside recursive node must survive');
  ensure(JSON.stringify(parseEnumNode(buildEnumNode_color('blue'))) === JSON.stringify({ variant: 'color', data: 'blue' }), 'standalone nonzero enum union must survive');
  const leaves: NestedLeaf[] = [{ variant: 'flag', data: true }, { variant: 'text', data: 'λ text payload' }];
  for (const leaf of leaves) {
    const nested: LeafParent = { variant: 'next', data: { variant: 'leaf', data: leaf } };
    ensure(JSON.stringify(parseLeafParent(buildLeafParent_next(nested))) === JSON.stringify({ variant: 'next', data: nested }), 'nested noncyclic union discriminant and payload must survive');
    ensure(JSON.stringify(parseLeafParent(buildLeafParent_leaf(leaf))) === JSON.stringify({ variant: 'leaf', data: leaf }), 'standalone struct union payload must survive');
  }

  // Real recursive canonical data: preserve ordered duplicate map keys, bool
  // bits, null vs omitted, empty containers, and nesting beyond old inline cap.
  let deep: Entry = { variant: 'str', data: 'λ\\quote"' };
  for (let depth = 0; depth < 8; depth++) deep = { variant: 'seq', data: [deep] };
  const entry: Entry = { variant: 'map', data: [
    { key: 'z', value: { variant: 'seq', data: [
      { variant: 'bool', data: true }, { variant: 'bool', data: false },
      { variant: 'number', data: 0.125 }, { variant: 'null', data: undefined }, deep,
    ] } },
    { key: 'z', value: { variant: 'map', data: [] } },
    { key: 'a', value: { variant: 'seq', data: [] } },
  ] };
  const criteria: NoulCriteria = {
    onTrue: { variant: 'some', data: entry },
    onFalse: { variant: 'none', data: undefined },
  };
  const decodedCriteria = parseNoulCriteria(buildNoulCriteria(criteria));
  ensure(JSON.stringify(decodedCriteria) === JSON.stringify(criteria), 'recursive Entry seq/map/struct/union must round trip without truncation');
  const absent = parseOptEntry(buildOptEntry_none());
  const explicitNull = parseOptEntry(buildOptEntry_some({ variant: 'null', data: undefined }));
  ensure(absent.variant === 'none', 'OptEntry.none must retain omitted-key semantics');
  ensure(explicitNull.variant === 'some' && explicitNull.data?.variant === 'null', 'OptEntry.some Entry.null must retain explicit null');


  // Real outer method and scoped method both execute generated typed checks.
  await remoteError(() => client.healthCheck(), 'DECISION_INPUT', 'outer input rejected');
  response = buildModelResponse_inferResult(1n, { variant: 'error', data: {
    message: 'cooperative deadline', code: 'DECISION_CANCELLED', details: '',
  } });
  await remoteError(() => client.infer('clef-flash-fp8:main').decide(request), 'DECISION_CANCELLED', 'cooperative deadline');
  const sent = calls[calls.length - 1];
  ensure(sent.service === 'model' && sent.method === 6, 'real decide must use canonical Model infer method metadata');
  const decoded = parseModelRequest(sent.bytes);
  ensure(decoded.variant === 'infer' && decoded.data.variant === 'decide', 'actual public request must select native decide');
  ensure(decoded.data.modelRef === 'clef-flash-fp8:main', 'curried model scope must survive');
  ensure(decoded.data.data.clefJson === request.clefJson, 'original JSON editor bytes/order/numbers must survive');
  ensure(decoded.data.data.maxStateTokens === 0 && decoded.data.data.timeoutMs === 17, 'explicit optional zero/value must survive');

  response = buildModelResponse_error(1n, { message: 'DECISION_PROFILE text is not a code', code: '', details: '' });
  await remoteError(() => client.healthCheck(), '', 'DECISION_PROFILE text is not a code');
  ensure(new RpcError({ message: 'DECISION_CANCELLED' }).code === undefined, 'constructor must not infer a code from message');
  ensure(new RpcError({ message: '', code: 'OPAQUE_NEW_CODE' }).message === '', 'empty wire message is preserved');
  ensure(new RpcError(null).message === 'RPC error', 'legacy absent-message fallback is preserved');

  transportFailure = new Error('DECISION_CANCELLED transport text is not a remote denial');
  try { await client.infer('clef-flash-fp8:main').decide(request); throw new Error('missing transport exception'); }
  catch (error) { ensure(error === transportFailure && !(error instanceof RpcError), 'transport failure must propagate untyped unchanged'); }
  transportFailure = undefined;
  // Truncate a REAL generated canonical frame; never invent a manual RPC encoder.
  response = buildModelResponse_error(1n, { message: 'real frame', code: 'DECISION_INPUT', details: '' }).slice(0, 3);
  let parserRejected = false;
  try { await client.healthCheck(); }
  catch (error) { parserRejected = true; ensure(error instanceof Error && !(error instanceof RpcError), 'parser failure must remain untyped'); }
  ensure(parserRejected, 'truncated actual frame must fail in the parser');

  // Actual multi-scope Registry client: a repository-level error must be
  // checked BEFORE unwrapping toward a worktree result. No backend is invoked.
  response = buildRegistryResponse_repoResult(1n, { variant: 'error', data: {
    message: 'intermediate repository denial', code: 'SCOPE_DENIED', details: '',
  } });
  const registry = new RegistryClient(transport);
  let streamRejected = false;
  try { await registry.getBlob({ grantRepo: 'wire-fixture-repo', variant: 'xetMerkle', data: 'wire-fixture-cid' }); }
  catch (error) { streamRejected = true; ensure(error === streamingFixtureFailure, 'stream transport failure remains untyped'); }
  ensure(streamRejected, 'actual Registry stream fixture must stop before delivery');
  const blobSent = calls[calls.length - 1];
  ensure(blobSent.service === 'registry' && blobSent.method === 9, 'actual getBlob preserves bound streaming method metadata');
  const blobRequest = parseRegistryRequest(blobSent.bytes);
  ensure(blobRequest.variant === 'getBlob' && blobRequest.data?.variant === 'xetMerkle' && blobRequest.data.data === 'wire-fixture-cid' && blobRequest.data.grantRepo === 'wire-fixture-repo', 'actual generated Registry client serializes the complete typed mixed union');
  await remoteError(() => registry.repo('wire-fixture-repo').worktree('main').stageAll(), 'SCOPE_DENIED', 'intermediate repository denial');
  const scoped = calls[calls.length - 1];
  ensure(scoped.service === 'registry' && scoped.method === 8, 'nested registry must preserve canonical bound metadata');

  for (const calib of [undefined, ''] as const) {
    const withPresenceBatch: DecisionBatch = { ...batch, version: { schema: 'wire-fixture-schema', model: 'wire-fixture-model', calib } };
    const withPresence: DecisionResponse = { ...decision, batch: withPresenceBatch };
    response = buildModelResponse_inferResult(1n, { variant: 'decide', data: withPresence });
    const actual = await client.infer('clef-flash-fp8:main').decide(request);
    ensure(JSON.stringify(actual) === JSON.stringify(withPresence), 'real decide response/float32/provenance/limits must round trip');
    const standalone = parseDecisionBatch(buildDecisionBatch(withPresenceBatch));
    ensure(JSON.stringify(standalone) === JSON.stringify(withPresenceBatch), 'actual data-only DecisionBatch import/codecs must work');
    ensure(actual.batch !== null, 'actual decide batch pointer must be populated');
    ensure(actual.batch.rows[0].answers[0].conformalSet === undefined, 'absent conformal remains absent');
    ensure(actual.batch.rows[0].answers[1].conformalSet?.length === 0, 'present-empty conformal remains present');
  }
}
void main().catch(error => { setTimeout(() => { throw error; }); });
