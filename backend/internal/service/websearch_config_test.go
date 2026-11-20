//go:build unit

package service

import (
	"context"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/pkg/websearch"
	"github.com/stretchr/testify/require"
)

// --- validateWebSearchConfig ---

func TestValidateWebSearchConfig_Nil(t *testing.T) {
	require.NoError(t, validateWebSearchConfig(nil))
}

func TestValidateWebSearchConfig_Valid(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Enabled: true,
		Providers: []WebSearchProviderConfig{
			{Type: "brave", QuotaLimit: int64Ptr(1000)},
			{Type: "tavily", QuotaLimit: int64Ptr(500)},
		},
	}
	require.NoError(t, validateWebSearchConfig(cfg))
}

func TestValidateWebSearchConfig_TooManyProviders(t *testing.T) {
	cfg := &WebSearchEmulationConfig{Providers: make([]WebSearchProviderConfig, maxWebSearchProviders+1)}
	for i := range cfg.Providers {
		cfg.Providers[i] = WebSearchProviderConfig{Type: "brave"}
	}
	err := validateWebSearchConfig(cfg)
	require.ErrorContains(t, err, "too many providers")
}

func TestValidateWebSearchConfig_InvalidType(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{{Type: "bing"}},
	}
	require.ErrorContains(t, validateWebSearchConfig(cfg), "invalid type")
}

func TestValidateWebSearchConfig_NegativeQuotaLimit(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{{Type: "brave", QuotaLimit: int64Ptr(-1)}},
	}
	require.ErrorContains(t, validateWebSearchConfig(cfg), "quota_limit must be > 0 or null")
}

func TestValidateWebSearchConfig_DuplicateTypeAllowed(t *testing.T) {
	// 同类型多 key 是受支持的多 key 负载均衡形态，不再按类型去重。
	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{
			{Type: "tavily"},
			{Type: "tavily"},
		},
	}
	require.NoError(t, validateWebSearchConfig(cfg))
}

func TestCheckWebSearchAPIKeys_MissingKey(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Enabled:   true,
		Providers: []WebSearchProviderConfig{{Type: "brave"}},
	}
	err := checkWebSearchAPIKeys(cfg)
	require.Error(t, err)
	require.Contains(t, err.Error(), "no API key configured")
}

func TestCheckWebSearchAPIKeys_DuplicateKey(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Enabled: true,
		Providers: []WebSearchProviderConfig{
			{Type: "tavily", APIKey: "tvly-1"},
			{Type: "tavily", APIKey: "tvly-1"},
		},
	}
	err := checkWebSearchAPIKeys(cfg)
	require.Error(t, err)
	require.Contains(t, err.Error(), "duplicate API key")
}

func TestCheckWebSearchAPIKeys_MultipleDistinctKeysOK(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Enabled: true,
		Providers: []WebSearchProviderConfig{
			{Type: "tavily", APIKey: "tvly-1"},
			{Type: "tavily", APIKey: "tvly-2"},
			{Type: "tavily", APIKey: "tvly-3"},
			{Type: "brave", APIKey: "brave-1"},
		},
	}
	require.NoError(t, checkWebSearchAPIKeys(cfg))
}

func TestCheckWebSearchAPIKeys_DisabledAllowsEmpty(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Enabled:   false,
		Providers: []WebSearchProviderConfig{{Type: "brave"}},
	}
	require.NoError(t, checkWebSearchAPIKeys(cfg))
}

func TestValidateWebSearchConfig_NilQuotaLimit(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{{Type: "brave", QuotaLimit: nil}},
	}
	require.NoError(t, validateWebSearchConfig(cfg))
}

// --- parseWebSearchConfigJSON ---

func TestParseWebSearchConfigJSON_ValidJSON(t *testing.T) {
	raw := `{"enabled":true,"providers":[{"type":"brave","api_key":"sk-xxx"}]}`
	cfg := parseWebSearchConfigJSON(raw)
	require.True(t, cfg.Enabled)
	require.Len(t, cfg.Providers, 1)
	require.Equal(t, "brave", cfg.Providers[0].Type)
}

func TestParseWebSearchConfigJSON_EmptyString(t *testing.T) {
	cfg := parseWebSearchConfigJSON("")
	require.False(t, cfg.Enabled)
	require.Empty(t, cfg.Providers)
}

func TestParseWebSearchConfigJSON_InvalidJSON(t *testing.T) {
	cfg := parseWebSearchConfigJSON("not{json")
	require.False(t, cfg.Enabled)
	require.Empty(t, cfg.Providers)
}

func TestParseWebSearchConfigJSON_BackwardCompatibility(t *testing.T) {
	// Old config with priority and quota_refresh_interval should parse without error
	raw := `{"enabled":true,"providers":[{"type":"brave","priority":1,"quota_refresh_interval":"monthly","quota_limit":1000}]}`
	cfg := parseWebSearchConfigJSON(raw)
	require.True(t, cfg.Enabled)
	require.Len(t, cfg.Providers, 1)
	require.Equal(t, int64(1000), *cfg.Providers[0].QuotaLimit)
}

// --- SanitizeWebSearchConfig ---

func TestSanitizeWebSearchConfig_MaskAPIKey(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Enabled: true,
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "sk-secret-xxx"},
		},
	}
	out := SanitizeWebSearchConfig(context.Background(), cfg)
	require.Equal(t, "", out.Providers[0].APIKey)
	require.True(t, out.Providers[0].APIKeyConfigured)
}

func TestSanitizeWebSearchConfig_NoAPIKey(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{{Type: "brave", APIKey: ""}},
	}
	out := SanitizeWebSearchConfig(context.Background(), cfg)
	require.Equal(t, "", out.Providers[0].APIKey)
	require.False(t, out.Providers[0].APIKeyConfigured)
}

func TestSanitizeWebSearchConfig_Nil(t *testing.T) {
	require.Nil(t, SanitizeWebSearchConfig(context.Background(), nil))
}

func TestSanitizeWebSearchConfig_PreservesOtherFields(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Enabled: true,
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "secret", QuotaLimit: int64Ptr(1000)},
		},
	}
	out := SanitizeWebSearchConfig(context.Background(), cfg)
	require.True(t, out.Enabled)
	require.Equal(t, int64(1000), *out.Providers[0].QuotaLimit)
}

func TestSanitizeWebSearchConfig_DoesNotMutateOriginal(t *testing.T) {
	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{{Type: "brave", APIKey: "secret"}},
	}
	_ = SanitizeWebSearchConfig(context.Background(), cfg)
	require.Equal(t, "secret", cfg.Providers[0].APIKey)
}

// --- PopulateWebSearchUsage ---

func TestPopulateWebSearchUsage_NilInput(t *testing.T) {
	require.Nil(t, PopulateWebSearchUsage(context.Background(), nil))
}

func TestPopulateWebSearchUsage_NoManager_QuotaUsedZero(t *testing.T) {
	// Ensure no global manager is set
	SetWebSearchManager(nil)
	defer SetWebSearchManager(nil)

	cfg := &WebSearchEmulationConfig{
		Enabled: true,
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "sk-key", QuotaLimit: int64Ptr(1000)},
		},
	}
	out := PopulateWebSearchUsage(context.Background(), cfg)
	require.NotNil(t, out)
	require.Len(t, out.Providers, 1)
	require.Equal(t, int64(0), out.Providers[0].QuotaUsed)
}

func TestPopulateWebSearchUsage_APIKeyConfigured_True(t *testing.T) {
	SetWebSearchManager(nil)
	defer SetWebSearchManager(nil)

	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "sk-key"},
		},
	}
	out := PopulateWebSearchUsage(context.Background(), cfg)
	require.True(t, out.Providers[0].APIKeyConfigured)
}

func TestPopulateWebSearchUsage_APIKeyConfigured_False(t *testing.T) {
	SetWebSearchManager(nil)
	defer SetWebSearchManager(nil)

	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: ""},
		},
	}
	out := PopulateWebSearchUsage(context.Background(), cfg)
	require.False(t, out.Providers[0].APIKeyConfigured)
}

func TestPopulateWebSearchUsage_NilQuotaLimit(t *testing.T) {
	SetWebSearchManager(nil)
	defer SetWebSearchManager(nil)

	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "sk-key", QuotaLimit: nil},
		},
	}
	out := PopulateWebSearchUsage(context.Background(), cfg)
	require.Nil(t, out.Providers[0].QuotaLimit)
}

func TestPopulateWebSearchUsage_NonNilQuotaLimit(t *testing.T) {
	SetWebSearchManager(nil)
	defer SetWebSearchManager(nil)

	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "sk-key", QuotaLimit: int64Ptr(500)},
		},
	}
	out := PopulateWebSearchUsage(context.Background(), cfg)
	require.NotNil(t, out.Providers[0].QuotaLimit)
	require.Equal(t, int64(500), *out.Providers[0].QuotaLimit)
}

func TestPopulateWebSearchUsage_WithManager_NilRedis(t *testing.T) {
	// Manager with nil Redis returns 0 usage without error
	mgr := websearch.NewManager([]websearch.ProviderConfig{
		{Type: "brave", APIKey: "k"},
	}, nil)
	SetWebSearchManager(mgr)
	defer SetWebSearchManager(nil)

	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "sk-key", QuotaLimit: int64Ptr(1000)},
		},
	}
	out := PopulateWebSearchUsage(context.Background(), cfg)
	require.Equal(t, int64(0), out.Providers[0].QuotaUsed)
	require.True(t, out.Providers[0].APIKeyConfigured)
}

func TestPopulateWebSearchUsage_DoesNotMutateOriginal(t *testing.T) {
	SetWebSearchManager(nil)
	defer SetWebSearchManager(nil)

	cfg := &WebSearchEmulationConfig{
		Providers: []WebSearchProviderConfig{
			{Type: "brave", APIKey: "secret", QuotaLimit: int64Ptr(100)},
		},
	}
	_ = PopulateWebSearchUsage(context.Background(), cfg)
	// Original should be unchanged
	require.Equal(t, "secret", cfg.Providers[0].APIKey)
	require.Equal(t, int64(0), cfg.Providers[0].QuotaUsed)
}

// --- ResetWebSearchUsage ---

func TestResetWebSearchUsage_NilManager(t *testing.T) {
	SetWebSearchManager(nil)
	defer SetWebSearchManager(nil)

	err := ResetWebSearchUsage(context.Background(), "brave", "")
	require.Error(t, err)
	require.Contains(t, err.Error(), "not initialized")
}
