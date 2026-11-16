package auth

import (
	"testing"
	"time"
)

func TestRoleHierarchy(t *testing.T) {
	admin := &Identity{Role: RoleAdmin}
	operator := &Identity{Role: RoleOperator}
	inference := &Identity{Role: RoleInference}
	auditor := &Identity{Role: RoleAuditor}

	// Admin satisfies all
	if !admin.HasRole(RoleAdmin) || !admin.HasRole(RoleOperator) || !admin.HasRole(RoleInference) || !admin.HasRole(RoleAuditor) {
		t.Fatalf("admin identity must satisfy all roles")
	}

	// Operator
	if operator.HasRole(RoleAdmin) {
		t.Fatalf("operator must not satisfy admin role")
	}
	if !operator.HasRole(RoleOperator) || !operator.HasRole(RoleInference) || !operator.HasRole(RoleAuditor) {
		t.Fatalf("operator must satisfy operator, inference, and auditor roles")
	}

	// Inference
	if inference.HasRole(RoleAdmin) || inference.HasRole(RoleOperator) {
		t.Fatalf("inference must not satisfy admin or operator roles")
	}
	if !inference.HasRole(RoleInference) || !inference.HasRole(RoleAuditor) {
		t.Fatalf("inference must satisfy inference and auditor roles")
	}

	// Auditor
	if auditor.HasRole(RoleAdmin) || auditor.HasRole(RoleOperator) || auditor.HasRole(RoleInference) {
		t.Fatalf("auditor must not satisfy admin, operator, or inference roles")
	}
	if !auditor.HasRole(RoleAuditor) {
		t.Fatalf("auditor must satisfy auditor role")
	}
}

func TestTokenLifecycleAndValidation(t *testing.T) {
	tm := NewTokenManager()

	// 1. Generate Admin token
	rawAdmin, recAdmin, err := tm.GenerateToken("cluster-admin", RoleAdmin, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate admin token: %v", err)
	}
	if recAdmin.Role != RoleAdmin {
		t.Fatalf("expected admin role, got %s", recAdmin.Role)
	}
	if len(rawAdmin) < 40 || !recAdmin.ExpiresAt.After(time.Now()) {
		t.Fatalf("invalid token format or expiration")
	}

	// 2. Validate valid token
	ident, err := tm.ValidateToken(rawAdmin)
	if err != nil {
		t.Fatalf("validate token: %v", err)
	}
	if ident.ID != recAdmin.ID || ident.Role != RoleAdmin || ident.Name != "cluster-admin" {
		t.Fatalf("unexpected validated identity: %+v", ident)
	}

	// 3. Reject bad tokens
	if _, err := tm.ValidateToken("abh_adm_invalid123"); err == nil {
		t.Fatalf("expected error for bogus token, got nil")
	}
	if _, err := tm.ValidateToken(""); err == nil {
		t.Fatalf("expected error for empty token, got nil")
	}

	// 4. Token expiration
	rawShort, _, err := tm.GenerateToken("short-lived", RoleInference, 5*time.Millisecond)
	if err != nil {
		t.Fatalf("generate short token: %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	if _, err := tm.ValidateToken(rawShort); err != ErrTokenExpired {
		t.Fatalf("expected ErrTokenExpired, got %v", err)
	}

	// 5. Token revocation
	rawRev, recRev, err := tm.GenerateToken("to-revoke", RoleOperator, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate rev token: %v", err)
	}
	if err := tm.Revoke(recRev.ID); err != nil {
		t.Fatalf("revoke token: %v", err)
	}
	if _, err := tm.ValidateToken(rawRev); err != ErrTokenRevoked {
		t.Fatalf("expected ErrTokenRevoked, got %v", err)
	}

	// 6. Token rotation
	rawRot, recRot, err := tm.GenerateToken("to-rotate", RoleAuditor, 1*time.Hour)
	if err != nil {
		t.Fatalf("generate rotate token: %v", err)
	}
	newRaw, newRec, err := tm.Rotate(recRot.ID, 2*time.Hour)
	if err != nil {
		t.Fatalf("rotate token: %v", err)
	}
	if newRaw == rawRot || newRec.ID == recRot.ID {
		t.Fatalf("rotated token must have new secret and ID")
	}
	// Old token must now be revoked
	if _, err := tm.ValidateToken(rawRot); err != ErrTokenRevoked {
		t.Fatalf("expected old token revoked, got %v", err)
	}
	// New token must validate
	newIdent, err := tm.ValidateToken(newRaw)
	if err != nil || newIdent.Role != RoleAuditor {
		t.Fatalf("validate new rotated token failed: %v", err)
	}
}

func TestStaticAdminTokenBinding(t *testing.T) {
	tm := NewTokenManager()
	staticSecret := "my-secret-cluster-bearer-token-12345"

	if err := tm.RegisterStaticAdminToken(staticSecret, "env-admin"); err != nil {
		t.Fatalf("register static admin token: %v", err)
	}

	ident, err := tm.ValidateToken(staticSecret)
	if err != nil {
		t.Fatalf("validate static admin token: %v", err)
	}
	if ident.Role != RoleAdmin || ident.Name != "env-admin" {
		t.Fatalf("unexpected static admin identity: %+v", ident)
	}
}
