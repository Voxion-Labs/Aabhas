// Package auth provides production identity, role-based access control (RBAC),
// cryptographic token management, and secure TLS transport configuration.
package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

// Role represents an authorization tier in the Aabhas control plane.
type Role string

const (
	// RoleAdmin has unrestricted access: engine lifecycle, token management, licenses, and config.
	RoleAdmin Role = "admin"
	// RoleOperator has operational access: hardware compatibility, benchmarking, and model inventory.
	RoleOperator Role = "operator"
	// RoleInference has execution access: token generation, tokenization, and model querying.
	RoleInference Role = "inference"
	// RoleAuditor has read-only access: metrics, health checks, and system diagnostics.
	RoleAuditor Role = "auditor"
)

// Standard errors for identity and authentication operations.
var (
	ErrInvalidToken     = errors.New("invalid or malformed authorization token")
	ErrTokenExpired     = errors.New("authorization token has expired")
	ErrTokenRevoked     = errors.New("authorization token has been revoked")
	ErrPermissionDenied = errors.New("insufficient role privileges for requested resource")
	ErrTokenNotFound    = errors.New("authorization token not found")
	ErrInvalidRole      = errors.New("invalid or unrecognized authorization role")
)

// Identity represents an authenticated caller identity.
type Identity struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Role      Role      `json:"role"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at,omitempty"`
}

// HasRole checks whether the identity satisfies the required role.
// RoleAdmin satisfies all role requirements.
// RoleOperator satisfies RoleOperator, RoleInference, and RoleAuditor.
// RoleInference satisfies RoleInference and RoleAuditor.
// RoleAuditor satisfies only RoleAuditor.
func (id *Identity) HasRole(required Role) bool {
	if id == nil {
		return false
	}
	if id.Role == RoleAdmin {
		return true
	}
	switch required {
	case RoleAdmin:
		return id.Role == RoleAdmin
	case RoleOperator:
		return id.Role == RoleOperator
	case RoleInference:
		return id.Role == RoleOperator || id.Role == RoleInference
	case RoleAuditor:
		return id.Role == RoleOperator || id.Role == RoleInference || id.Role == RoleAuditor
	default:
		return id.Role == required
	}
}

// TokenRecord holds persistent token metadata with cryptographic hash (raw token is never stored).
type TokenRecord struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Role      Role      `json:"role"`
	TokenHash string    `json:"token_hash"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at,omitempty"`
	Revoked   bool      `json:"revoked"`
}

// TokenManager manages cryptographic tokens, issuance, constant-time validation, and rotation.
type TokenManager struct {
	mu     sync.RWMutex
	tokens map[string]*TokenRecord // keyed by token ID
	byHash map[string]*TokenRecord // keyed by SHA-256 hex hash
}

// NewTokenManager constructs an in-memory thread-safe cryptographic token manager.
func NewTokenManager() *TokenManager {
	return &TokenManager{
		tokens: make(map[string]*TokenRecord),
		byHash: make(map[string]*TokenRecord),
	}
}

// GenerateToken creates a new cryptographic token with prefix, 32 bytes of CSPRNG entropy, and returns the raw token once.
func (tm *TokenManager) GenerateToken(name string, role Role, ttl time.Duration) (string, *TokenRecord, error) {
	if !isValidRole(role) {
		return "", nil, fmt.Errorf("%w: %s", ErrInvalidRole, role)
	}
	name = strings.TrimSpace(name)
	if name == "" {
		name = "unnamed-service"
	}

	randomBytes := make([]byte, 32)
	if _, err := rand.Read(randomBytes); err != nil {
		return "", nil, fmt.Errorf("generate token entropy: %w", err)
	}

	var prefix string
	switch role {
	case RoleAdmin:
		prefix = "abh_adm_"
	case RoleOperator:
		prefix = "abh_ops_"
	case RoleInference:
		prefix = "abh_inf_"
	case RoleAuditor:
		prefix = "abh_aud_"
	default:
		prefix = "abh_key_"
	}

	rawToken := prefix + hex.EncodeToString(randomBytes)
	hash := hashToken(rawToken)
	tokenID := fmt.Sprintf("tok-%s", hex.EncodeToString(randomBytes[:8]))

	now := time.Now().UTC()
	var expiresAt time.Time
	if ttl > 0 {
		expiresAt = now.Add(ttl)
	}

	record := &TokenRecord{
		ID:        tokenID,
		Name:      name,
		Role:      role,
		TokenHash: hash,
		CreatedAt: now,
		ExpiresAt: expiresAt,
		Revoked:   false,
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.tokens[tokenID] = record
	tm.byHash[hash] = record

	return rawToken, record, nil
}

// RegisterStaticAdminToken binds an existing environment token (such as AABHAS_API_TOKEN) as an admin identity.
func (tm *TokenManager) RegisterStaticAdminToken(rawToken string, name string) error {
	rawToken = strings.TrimSpace(rawToken)
	if rawToken == "" {
		return nil
	}
	hash := hashToken(rawToken)

	tm.mu.Lock()
	defer tm.mu.Unlock()

	// If already registered, skip
	if _, exists := tm.byHash[hash]; exists {
		return nil
	}

	tokenID := "tok-static-admin"
	record := &TokenRecord{
		ID:        tokenID,
		Name:      name,
		Role:      RoleAdmin,
		TokenHash: hash,
		CreatedAt: time.Now().UTC(),
		Revoked:   false,
	}
	tm.tokens[tokenID] = record
	tm.byHash[hash] = record
	return nil
}

// ValidateToken validates a raw bearer token against active records using constant-time hash comparison.
func (tm *TokenManager) ValidateToken(rawToken string) (*Identity, error) {
	rawToken = strings.TrimSpace(rawToken)
	if rawToken == "" {
		return nil, ErrInvalidToken
	}

	computedHash := hashToken(rawToken)

	tm.mu.RLock()
	defer tm.mu.RUnlock()

	// Locate candidate record
	record, exists := tm.byHash[computedHash]
	if !exists {
		// Constant-time dummy comparison to mitigate timing attacks
		var dummy [32]byte
		subtle.ConstantTimeCompare([]byte(computedHash), dummy[:])
		return nil, ErrInvalidToken
	}

	// Constant-time hash check
	if subtle.ConstantTimeCompare([]byte(record.TokenHash), []byte(computedHash)) != 1 {
		return nil, ErrInvalidToken
	}

	if record.Revoked {
		return nil, ErrTokenRevoked
	}

	if !record.ExpiresAt.IsZero() && time.Now().UTC().After(record.ExpiresAt) {
		return nil, ErrTokenExpired
	}

	return &Identity{
		ID:        record.ID,
		Name:      record.Name,
		Role:      record.Role,
		CreatedAt: record.CreatedAt,
		ExpiresAt: record.ExpiresAt,
	}, nil
}

// Revoke marks a token as revoked by ID.
func (tm *TokenManager) Revoke(tokenID string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	record, exists := tm.tokens[tokenID]
	if !exists {
		return ErrTokenNotFound
	}
	record.Revoked = true
	return nil
}

// Rotate invalidates an existing token and creates a replacement with identical name and role.
func (tm *TokenManager) Rotate(tokenID string, newTTL time.Duration) (string, *TokenRecord, error) {
	tm.mu.Lock()
	record, exists := tm.tokens[tokenID]
	if !exists {
		tm.mu.Unlock()
		return "", nil, ErrTokenNotFound
	}
	record.Revoked = true
	name := record.Name
	role := record.Role
	tm.mu.Unlock()

	return tm.GenerateToken(name, role, newTTL)
}

// ListTokens returns a snapshot of all registered token metadata.
func (tm *TokenManager) ListTokens() []TokenRecord {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	list := make([]TokenRecord, 0, len(tm.tokens))
	for _, rec := range tm.tokens {
		list = append(list, *rec)
	}
	return list
}

// BuildTLSConfig constructs a modern TLS 1.3/1.2 server configuration with optional mutual TLS (mTLS).
func BuildTLSConfig(certFile, keyFile, clientCAFile string, requireClientCert bool) (*tls.Config, error) {
	if certFile == "" || keyFile == "" {
		return nil, errors.New("TLS requires both cert_file and key_file")
	}

	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("load TLS key pair: %w", err)
	}

	cfg := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_CHACHA20_POLY1305_SHA256,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
		},
	}

	if clientCAFile != "" {
		caData, err := os.ReadFile(clientCAFile)
		if err != nil {
			return nil, fmt.Errorf("read client CA file: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caData) {
			return nil, errors.New("no valid certificates found in client CA file")
		}
		cfg.ClientCAs = pool
		if requireClientCert {
			cfg.ClientAuth = tls.RequireAndVerifyClientCert
		} else {
			cfg.ClientAuth = tls.VerifyClientCertIfGiven
		}
	}

	return cfg, nil
}

func hashToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

func isValidRole(r Role) bool {
	switch r {
	case RoleAdmin, RoleOperator, RoleInference, RoleAuditor:
		return true
	default:
		return false
	}
}
