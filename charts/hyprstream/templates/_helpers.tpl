{{/*
Chart name (overridable).
*/}}
{{- define "hyprstream.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name.
*/}}
{{- define "hyprstream.fullname" -}}
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

{{- define "hyprstream.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "hyprstream.labels" -}}
helm.sh/chart: {{ include "hyprstream.chart" . }}
{{ include "hyprstream.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels (chart-wide).
*/}}
{{- define "hyprstream.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hyprstream.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ServiceAccount name.
*/}}
{{- define "hyprstream.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "hyprstream.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Per-service object name: <fullname>-<serviceName>.
Usage: include "hyprstream.svcName" (dict "root" $ "name" $name)
*/}}
{{- define "hyprstream.svcName" -}}
{{- printf "%s-%s" (include "hyprstream.fullname" .root) .name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Resolve the container image for a service, applying the per-service variant
suffix. Usage: include "hyprstream.image" (dict "root" $ "svc" $svc)
*/}}
{{- define "hyprstream.image" -}}
{{- $img := .root.Values.image -}}
{{- $tag := default .root.Chart.AppVersion $img.tag -}}
{{- $variant := .svc.variant | default "" -}}
{{- $suffix := "" -}}
{{- if $variant -}}
{{- $suffix = index $img.variants $variant | default "" -}}
{{- end -}}
{{- printf "%s:%s%s" $img.repository $tag $suffix -}}
{{- end }}
