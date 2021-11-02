package logger

import (
	"io"
	"log"
	"os"
)

var (
	// Debug is a logger for debug level messages
	Debug = log.New(os.Stdout, "[Debug] Regression: ", log.Lshortfile)
	// Info is a logger for infomation level messages
	Info = log.New(os.Stdout, "[Info] Regression: ", 0)
	// Warn is a logger for warning level messages
	Warn = log.New(os.Stderr, "[Warning] Regression: ", 0)
	// Err is a logger for error level messages
	Err     = log.New(os.Stderr, "[Error] Regression: ", 0)
	loggers = []*log.Logger{Debug, Info, Warn, Err}
)

// SetLogsFlags : すべての種類のログに設定を適用する
func SetLogsFlags(flags int) {
	for _, logger := range loggers {
		logger.SetFlags(flags)
	}
}

// SetLogsOutput : すべての種類のログの出力先を変更する
func SetLogsOutput(w io.Writer) {
	for _, logger := range loggers {
		logger.SetOutput(w)
	}
}

// SetLogsPrefix : すべての種類のログメッセージのPrefixを設定する
func SetLogsPrefix(prefix string) {
	for _, logger := range loggers {
		logger.SetPrefix(prefix)
	}
}
