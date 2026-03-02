# Sync from yzma

Script untuk sync packages dari upstream [yzma](https://github.com/hybridgroup/yzma) ke llamalib.

## Packages yang di-sync

Script ini men-sync packages berikut dari yzma:

- `llama` - Core llama.cpp bindings
- `loader` - Model loader utilities
- `message` - Chat message and tool call types
- `mtmd` - Multimodal (vision/audio) support
- `template` - Chat template (Jinja) support
- `utils` - Platform utilities

## Packages yang TIDAK di-sync

Packages berikut **tidak** di-sync karena memiliki custom changes:

- `download` - Menggunakan `kawai-network/grab` (bukan `hashicorp/go-getter` seperti yzma)
- `installer*` - Custom installer logic untuk llamalib
- `model_specs.go` - Custom model specifications
- `service.go` - Custom service layer (jika ada)
- `toolcall.go` - Custom tool call handling
- `templates.go` - Custom template helpers

## Prerequisites

1. **Repo yzma harus ada di lokal**

   Clone atau pull repo yzma terlebih dahulu:
   ```bash
   cd ~/github.com/kawai-network
   git clone https://github.com/hybridgroup/yzma.git
   # atau jika sudah ada
   cd yzma && git pull
   ```

2. **rsync harus terinstall** (sudah ada di macOS)

## Cara Pakai

### Basic Usage

```bash
cd /Users/yuda/github.com/kawai-network/llamalib
./scripts/sync-from-yzma.sh ../yzma
```

### Dengan Path Custom

```bash
./scripts/sync-from-yzma.sh /path/to/yzma
```

## Workflow

### 1. Run Sync

```bash
./scripts/sync-from-yzma.sh ../yzma
```

Script akan:
- Copy files dari `yzma/pkg/{llama,loader,message,mtmd,template,utils}/` ke `{llama,loader,message,mtmd,template,utils}/`
- Replace import paths: `github.com/hybridgroup/yzma/pkg/` → `github.com/getkawai/llamalib/`
- Delete files di llamalib yang tidak ada di yzma

### 2. Review Changes

```bash
git status
git diff
```

### 3. Run Tests

```bash
go build ./...
go test ./llama/... ./mtmd/... ./message/...
```

### 4. Commit

```bash
git add llama loader message mtmd template utils scripts
git commit -m "sync: update from yzma@<commit-hash>

- llama: sync from yzma pkg/llama
- mtmd: sync from yzma pkg/mtmd
- loader, message, template, utils: sync latest changes
- fix: update FFI types to match yzma (FFIType* -> ffiType*)"
```

## Frekuensi Sync

Disarankan sync **setiap 1-2 minggu** atau setelah ada perubahan penting di yzma:

- New llama.cpp bindings
- Bug fixes
- Performance improvements
- API changes

Cek changelog yzma:
```bash
cd ../yzma
git log --oneline --since="2 weeks ago" -- pkg/llama pkg/mtmd
```

## Troubleshooting

### Build Error: Import Path Not Found

Jika ada error import path setelah sync:
```bash
# Cari file yang masih pakai import path lama
grep -r "github.com/hybridgroup/yzma/pkg/" llama/ mtmd/ message/ loader/ template/ utils/

# Fix manual atau re-run script
./scripts/sync-from-yzma.sh ../yzma
```

### Conflict di Download Package

Package `download` tidak di-sync karena menggunakan dependency yang berbeda:
- llamalib: `github.com/kawai-network/grab`
- yzma: `github.com/hashicorp/go-getter`

Jika butuh fitur baru dari download yzma, merge manual:
1. Copy logic yang dibutuhkan
2. Adaptasi untuk pakai `kawai-network/grab`
3. Test thoroughly

### Test Fails

Jika test gagal setelah sync:
```bash
# Run specific package test
go test ./llama/... -v

# Check if it's a known issue in yzma
cd ../yzma
go test ./pkg/llama/... -v
```

## Version Tracking

Track commit yzma yang terakhir di-sync di commit message:

```
sync: update from yzma@33bee9dbd3849d73c04be77354dda1c91da7d8dc
```

Untuk cek perubahan sejak sync terakhir:
```bash
# Get last sync commit
LAST_SYNC=$(git log --oneline --grep="sync: update from yzma" | head -1 | awk '{print $NF}')

# Show changes in yzma since then
cd ../yzma
git log --oneline $LAST_SYNC..HEAD -- pkg/llama pkg/mtmd
```

## Alternative: Manual Sync

Jika script tidak bisa dipakai, sync manual dengan rsync:

```bash
# Sync llama package
rsync -av --delete \
  --exclude='go.mod' --exclude='go.sum' \
  ../yzma/pkg/llama/ ./llama/

# Replace import paths
find ./llama -name "*.go" -type f -exec sed -i.bak 's|github.com/hybridgroup/yzma/pkg/|github.com/getkawai/llamalib/|g' {} \;
find ./llama -name "*.bak" -delete
```

## History

- **Feb 5, 2026**: Last sync dari yzma commit `e406423` (masih pakai `FFIType*` uppercase)
- **Feb 5, 2026**: yzma commit `c1db16f` mengubah ke `ffiType*` (lowercase)
- **Mar 1, 2026**: Script sync dibuat untuk automate future updates

## References

- yzma upstream: https://github.com/hybridgroup/yzma
- yzma download package: https://github.com/hybridgroup/yzma/tree/main/pkg/download
- llamalib download package: https://github.com/getkawai/llamalib/tree/main/download
