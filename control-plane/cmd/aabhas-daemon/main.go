package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Voxion-Labs/Aabhas/control-plane/internal/api"
	"github.com/Voxion-Labs/Aabhas/control-plane/internal/cgo_bridge"
)

const shutdownTimeout = 15 * time.Second

func main() {
	logger := log.New(os.Stderr, "aabhas-daemon ", log.LstdFlags|log.LUTC)
	addr := os.Getenv("AABHAS_LISTEN_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}

	server := api.NewServer(addr, logger)
	// The native stop call is deferred before the server starts so every normal
	// shutdown path releases native mappings, including listen failures.
	defer func() {
		if err := cgo_bridge.Stop(); err != nil {
			logger.Printf("native engine stop during shutdown failed: %v", err)
		}
	}()

	serverErrors := make(chan error, 1)
	go func() {
		serverErrors <- server.ListenAndServe()
	}()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(signals)

	select {
	case sig := <-signals:
		logger.Printf("received shutdown signal %s", sig)
	case err := <-serverErrors:
		if !errors.Is(err, http.ErrServerClosed) {
			logger.Printf("REST server stopped unexpectedly: %v", err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		logger.Printf("REST server shutdown failed: %v", err)
	}
}
