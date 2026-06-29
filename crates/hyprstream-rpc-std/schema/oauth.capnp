@0xc1d2e3f4a5b60718;

# Cap'n Proto schema for OAuth user management service
#
# Provides user CRUD via ZMQ RPC for CLI and SCIM transports.
# Uses REQ/REP pattern.

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".scope;
using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".optional;

struct OauthRequest {
  id @0 :UInt64;

  union {
    registerUser @1 :RegisterUser
      $scope(manage) $mcpDescription("Register a new user");
    getUser @2 :Text
      $scope(query) $mcpDescription("Get user profile by username");
    listUsers @3 :ListUsers
      $scope(query) $mcpDescription("List/search users with optional filter");
    updateUser @4 :UpdateUser
      $scope(manage) $mcpDescription("Update user profile fields");
    suspendUser @5 :Text
      $scope(manage) $mcpDescription("Suspend user (set active=false)");
    resumeUser @6 :Text
      $scope(manage) $mcpDescription("Resume suspended user (set active=true)");
    removeUser @7 :Text
      $scope(manage) $mcpDescription("Permanently remove user");
    addPubkey @8 :AddPubkey
      $scope(manage) $mcpDescription("Add an Ed25519 public key to a user");
    removePubkey @9 :RemovePubkey
      $scope(manage) $mcpDescription("Remove a public key by fingerprint");
    listPubkeys @10 :Text
      $scope(query) $mcpDescription("List all public keys for a user");
  }
}

struct OauthResponse {
  requestId @0 :UInt64;

  union {
    error @1 :ErrorInfo;
    registerUserResult @2 :UserInfo;
    getUserResult @3 :UserInfo;
    listUsersResult @4 :UserListResult;
    updateUserResult @5 :UserInfo;
    suspendUserResult @6 :Void;
    resumeUserResult @7 :Void;
    removeUserResult @8 :Void;
    addPubkeyResult @9 :PubkeyEntry;
    removePubkeyResult @10 :Bool;
    listPubkeysResult @11 :List(PubkeyEntry);
  }
}

struct RegisterUser {
  username @0 :Text;
  pubkeyBase64 @1 :Text $optional;
  name @2 :Text $optional;
  email @3 :Text $optional;
  externalId @4 :Text $optional;
}

struct AddPubkey {
  username @0 :Text;
  pubkeyBase64 @1 :Text;
  label @2 :Text $optional;
}

struct RemovePubkey {
  username @0 :Text;
  fingerprint @1 :Text;
}

struct ListUsers {
  filter @0 :Text;
  activeOnly @1 :Bool;
  count @2 :UInt32;
  startIndex @3 :UInt32;
}

struct UpdateUser {
  username @0 :Text;
  name @1 :Text $optional;
  email @2 :Text $optional;
  externalId @3 :Text $optional;
  active @4 :Bool;
}

struct UserInfo {
  username @0 :Text;
  sub @1 :Text;
  pubkeyBase64 @2 :Text;  # DEPRECATED: use pubkeys list for new code
  name @3 :Text;
  email @4 :Text;
  emailVerified @5 :Bool;
  active @6 :Bool;
  externalId @7 :Text;
  pubkeys @8 :List(PubkeyEntry);  # Multiple pubkeys per user
}

struct PubkeyEntry {
  fingerprint @0 :Text;   # base64url SHA-256 of pubkey bytes
  pubkeyBase64 @1 :Text;  # base64url of raw 32-byte Ed25519 pubkey
  label @2 :Text;         # user-provided label (e.g., "laptop", "work")
  createdAt @3 :Int64;    # unix timestamp
  lastUsedAt @4 :Int64;   # unix timestamp, 0 if never used
}

struct UserListResult {
  users @0 :List(UserInfo);
  totalResults @1 :UInt32;
}
