# Archived native head finite-load evidence

This directory contains the **exact nine public inputs** used by
`crates/hyprstream-decision-head/fixtures/derive-native-head-finite-load.py`:
seven public receipts and two historical source snapshots. A fresh checkout
can reproduce the committed receipt using only Python's standard library:

```sh
python3 -S crates/hyprstream-decision-head/fixtures/derive-native-head-finite-load.py \
  --output /tmp/clef-head-finite-receipt-NEW.json
cmp /tmp/clef-head-finite-receipt-NEW.json \
  crates/hyprstream-decision-head/fixtures/native-head-finite-load-receipt.json
```

Choose a new output filename: the command deliberately refuses overwrites.
The optional `--evidence-root` and `--source-root` defaults are respectively
this directory's `public-receipts` and `source-snapshot`, resolved from the
script location, independently of the working directory. To test the package:

```sh
python3 -S crates/hyprstream-decision-head/fixtures/test_native_head_finite_load_evidence.py
```

The archived source **composition identity** is
`f2a82911ea8b6344db43ee1587760376a795832d74728edca0341ae3deb42a0b`;
it is not a Git commit. The **executable SHA256** is
`e90affaa7a958b9b1528eae99a6a8882ff02314010cc378491c6195865bdadb5`;
it is not a Git commit and its binary is not bundled or freshly hashed here.
The receipt records the previously executed development native load of all
122 tensors of head SHA256
`19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0`.
No model, tensor scan, inference, installation, network, private logs or
credentials are needed by this derivation.

The two `.rs` files are historical evidence data, **not compiled modules** or
current implementations. `source-snapshot/crates/hyprstream-decision-head/src/clef/head.rs`
retains its originating Apache-2.0 component license;
`source-snapshot/crates/hyprstream-inference/src/runtime/clef_decision.rs`
retains its originating AGPL-3.0-only component license. They are outside the
permissive crate. Their bytes and the seven receipts are unchanged so all nine
original source/evidence pins remain independently checkable.

Historical receipt paths and opaque hashes identify development artifacts;
they do not require retrieving the referenced private files. The derivation
opens only the seven included receipts and two included source files. In
particular, environment/private-log hashes are identifiers, not their bytes.
The included model responses are the previously declared public synthetic
five-record corpus, not user input. No signing authority or credential is
conferred by these records.

This is **archived development load-time evidence**. It is not a standalone
raw head scan, whole-backbone finite audit, current-main/new-candidate
qualification, task accuracy/calibration, native release, image or lab gate.
The separate 200-scale audit covers only scales. The normative finite-load
requirement and all model pins remain unchanged.
