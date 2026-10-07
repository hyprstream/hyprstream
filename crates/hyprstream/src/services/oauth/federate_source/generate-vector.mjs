// Offline test vector generator. npm dependencies are tooling only:
// @noble/post-quantum 0.7.1; @noble/curves 2.4.0.
// node generate-vector.mjs /path/to/tooling/node_modules
import { createHash, sign } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const root = process.argv[2];
const { ml_dsa65 } = await import(pathToFileURL(`${root}/@noble/post-quantum/ml-dsa.js`));
const { ed25519 } = await import(pathToFileURL(`${root}/@noble/curves/ed25519.js`));
const b = x => Buffer.from(x);
const s = x => b(x);
const sha = x => createHash('sha256').update(x).digest();
const b64 = x => b(x).toString('base64url');
const u32 = n => { const x=Buffer.alloc(4); x.writeUInt32BE(n); return x; };
const u64 = n => { const x=Buffer.alloc(8); x.writeBigUInt64BE(BigInt(n)); return x; };
const frame = (d, fs) => Buffer.concat([s(d), b([0]), u32(fs.length), ...fs.flatMap(x=>[u32(x.length),b(x)])]);
const F='https://login.federate.to', C='cyberdione-www-staging', W='https://www.staging.lab.hyprstream.com', H='https://discovery.staging.lab.hyprstream.com';
const edSeed=Buffer.alloc(32,7), pqSeed=Buffer.alloc(32,8);
const ed=ed25519.getPublicKey(edSeed), pq=ml_dsa65.keygen(pqSeed);
const state=b64(Buffer.alloc(32,2)), verifier=b64(Buffer.alloc(32,3)), pkce=b64(sha(s(verifier)));
const N=frame('hyprstream.federate.session-primary.nonce.v1',[s(F),s(C),s(W),s(W+'/federate/callback'),s(H),s(H),Buffer.alloc(32,1),ed,pq.publicKey,sha(s(state)),s(pkce)]);
const nonce='hsn1.'+b64(sha(N)), code='TEST-CODE-NOT-LIVE';
const header={alg:'RS256',typ:'JWT',kid:'fixture'};
const claims={iss:F,aud:C,sub:'opaque-sub',jti:'source-id',iat:1800000000,exp:1800000300,nonce,c_hash:b64(sha(s(code)).subarray(0,16)),federated_claims:{connector_id:'atproto',user_id:'did:plc:abcdefghijklmnopqrstuvwx'}};
const signed=b64(s(JSON.stringify(header)))+'.'+b64(s(JSON.stringify(claims)));
const token=signed+'.'+b64(sign('RSA-SHA256',s(signed),readFileSync(new URL('test-rsa.pem',import.meta.url))));
const kid=(a,k)=>sha(frame('hyprstream.session-primary.kid.v1',[s(a),k]));
const T=frame('hyprstream.session-primary.possession.v1',[s(H),s(H+'/oauth/token'),s(F),s(C),s(W),Buffer.alloc(32,9),sha(s(token)),s(nonce),sha(s(code)),sha(N),s('test-local-account'),s('test-rpc-subject'),s('test-tenant'),s(H),s('query:registry:*'),s('query:registry:*'),u64(1800000000),u64(1800000060),s('hs-cose-sign-ed25519-mldsa65-wns-v1'),kid('Ed25519',ed),kid('ML-DSA-65',pq.publicKey),s('test-grant-revision')]);
const es=ed25519.sign(T,edSeed), ps=ml_dsa65.sign(T,pq.secretKey,{extraEntropy:false});
if (!ed25519.verify(es,T,ed) || !ml_dsa65.verify(ps,T,pq.publicKey)) throw Error('self verification failed');
console.log(JSON.stringify({purpose:'public deterministic test seeds; no live credentials',ed_public:b64(ed),pq_public:b64(pq.publicKey),nonce,token,transcript:b64(T),ed_signature:b64(es),pq_signature:b64(ps)},null,2));
