{{/*
Sanitize name for environment variable usage
*/}}
{{- define "rag-lsd.envVarName" -}}
{{- . | replace " " "_" | replace "-" "_" | replace "." "_" | replace "/" "_" | replace ":" "_" | upper -}}
{{- end -}}

{{/*
Combine localModels and remoteModels into a single list.
This allows localModels to be in values.yaml (git) and remoteModels in values-secrets.yaml (not in git).
Wrapped in "items" key because fromYaml doesn't handle root-level lists.
*/}}
{{- define "allModels" -}}
{{- $local := .Values.localModels | default list -}}
{{- $remote := .Values.remoteModels | default list -}}
{{- dict "items" (concat $local $remote) | toYaml -}}
{{- end -}}

{{/*
Expand the name of the chart.
*/}}
{{- define "rag-lsd.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "rag-lsd.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}