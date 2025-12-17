# JWT Authentication for External Gateway Integration

**Status:** deprecated
**Date:** 2025-12-17
**Author:** Antigravity AI


---

## Context

With the adoption of the decoupled IBM ContextForge Gateway (ADR 058), we need a secure authentication mechanism for Project Sanctuary clients to communicate with the external gateway service running on localhost:4444.

**Requirements:**
- Secure authentication without storing secrets in the repository
- Support for asymmetric cryptography (public/private key pairs)
- Token-based authentication compatible with the gateway's security model
- Testability (ability to generate tokens for integration tests)

**Alternatives Considered:**
1. **Simple Bearer Token**: Easy but less secure, tokens are static and harder to rotate
2. **API Keys**: Similar security profile to bearer tokens
3. **JWT with RS256**: Industry standard, supports key rotation, verifiable signatures

## Decision

We will use **JWT (JSON Web Tokens) with RS256 algorithm** for authentication between Project Sanctuary and the external Gateway service.

**Implementation:**
- **Algorithm**: RS256 (RSA Signature with SHA-256)
- **Key Storage**: Private/public key pairs stored in `certs/gateway/` (gitignored)
- **Token Generation**: Tests generate JWTs using PyJWT library with the private key
- **Token Verification**: Gateway verifies JWTs using the public key
- **Key Distribution**: Keys are manually copied from the gateway's `certs/jwt/` directory

**Configuration:**
```bash
MCP_GATEWAY_JWT_PUBLIC_KEY_PATH=certs/gateway/public.pem
MCP_GATEWAY_JWT_PRIVATE_KEY_PATH=certs/gateway/private.pem
MCP_GATEWAY_JWT_ALGORITHM=RS256
```

**Security Measures:**
- Keys are excluded from Git (`.gitignore`)
- Keys are excluded from snapshot scripts
- Keys are excluded from RAG Cortex ingestion
- Tokens have 5-minute expiry for tests

## Consequences

**Positive:**
- Industry-standard authentication mechanism
- Asymmetric cryptography enables key rotation without updating clients
- Tokens are self-contained and verifiable
- No shared secrets stored in repository
- Compatible with OAuth 2.0 / OpenID Connect if needed in future

**Negative:**
- Requires PyJWT dependency for test suite
- Key management overhead (manual copy from gateway)
- More complex than simple bearer tokens
- Debugging token issues requires JWT inspection tools

**Risks:**
- Key leakage if not properly gitignored
- Clock skew between client and gateway could cause token validation failures
- Payload mismatch if gateway expects specific claims we don't provide


---

**Status Update (2025-12-17):** Gateway uses API token authentication instead of JWT. Tokens are created via the gateway's admin interface (/admin/tokens) rather than client-side JWT generation.
