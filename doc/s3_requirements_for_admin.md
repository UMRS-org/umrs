# UMRS S3 Storage Requirements for System Administrators
# UMRS S3ストレージ要件（システム管理者向け）

> Target S3 implementation: **Lens3** (https://github.com/RIKEN-RCCS/lens3)
>
> 対象S3実装: **Lens3** (https://github.com/RIKEN-RCCS/lens3)

---

## 1. Overview / 概要

UMRS (Unified Model & Resource Server) uses S3-compatible storage for:
- Storing user-uploaded data (images, text files)
- Storing inference results (segmentation masks, reconstructed images, feature tensors)
- Exchanging data between pipeline steps (intermediate staging objects)
- Generating presigned URLs for client-side direct upload/download

UMRSは以下の目的でS3互換ストレージを使用します：
- ユーザーがアップロードするデータ（画像、テキストファイル）の保存
- 推論結果（セグメンテーションマスク、再構成画像、特徴量テンソル）の保存
- パイプラインステップ間のデータ交換（中間ステージングオブジェクト）
- クライアント側での直接アップロード/ダウンロード用のPresigned URL生成

### Protocol & Client Library / プロトコルとクライアントライブラリ

- **Protocol**: S3 API (AWS S3 compatible) over HTTP/HTTPS
- **Client**: Python `boto3` (AWS SDK for Python)
- **Signature**: `s3v4` (Signature Version 4)
- **Addressing**: Path-style (`http://host:port/bucket/key`), **not** virtual-hosted-style

- **プロトコル**: S3 API (AWS S3互換) HTTP/HTTPS経由
- **クライアント**: Python `boto3` (AWS SDK for Python)
- **署名方式**: `s3v4` (署名バージョン4)
- **アドレッシング**: パス形式 (`http://host:port/bucket/key`)、仮想ホスト形式は**不使用**

---

## 2. Required S3 API Operations / 必要なS3 API操作

### 2.1 Bucket Operations / バケット操作

| S3 API Action | Usage / 用途 | Priority / 優先度 |
|---|---|---|
| `ListBuckets` | Connection validation, server type detection / 接続確認、サーバ種別検出 | **Required / 必須** |
| `HeadBucket` | Check if a bucket exists / バケット存在確認 | **Required / 必須** |
| `CreateBucket` | Create per-user and per-experiment buckets / ユーザー・実験ごとのバケット作成 | **Required / 必須** |
| `ListObjectsV2` | List objects in a bucket (with pagination) / バケット内オブジェクト一覧（ページネーション対応） | **Required / 必須** |
| `GetBucketLocation` | Determine bucket region (used implicitly by boto3) / バケットリージョン取得（boto3が暗黙的に使用） | **Required / 必須** |

### 2.2 Object Operations / オブジェクト操作

| S3 API Action | Usage / 用途 | Priority / 優先度 |
|---|---|---|
| `GetObject` | Download images, text, model data, pipeline data / 画像・テキスト・モデルデータ・パイプラインデータのダウンロード | **Required / 必須** |
| `PutObject` | Upload text, images, inference results, embeddings / テキスト・画像・推論結果・埋め込みベクトルのアップロード | **Required / 必須** |
| `HeadObject` | Check if an object exists (deduplication) / オブジェクト存在確認（重複排除） | **Required / 必須** |
| `DeleteObject` / `DeleteObjects` | Clean up intermediate staging objects after pipeline runs / パイプライン実行後の中間ステージングオブジェクト削除 | **Required / 必須** |

### 2.3 High-Level Operations (internally use above APIs) / 高レベル操作（内部的に上記APIを使用）

| boto3 Method | Translates to / 内部API | Usage / 用途 |
|---|---|---|
| `upload_file()` | `PutObject` (multipart for large files) | Upload local files / ローカルファイルのアップロード |
| `download_file()` | `GetObject` | Download to local path / ローカルパスへのダウンロード |
| `get_paginator("list_objects_v2")` | `ListObjectsV2` | Paginated listing / ページネーション付き一覧 |

### 2.4 Presigned URL Operations / Presigned URL操作

| Operation | Usage / 用途 | Priority / 優先度 |
|---|---|---|
| `generate_presigned_url` (GET) | Client-side direct download / クライアント側直接ダウンロード | **Required / 必須** |
| `generate_presigned_url` (PUT) | Client-side direct upload / クライアント側直接アップロード | **Required / 必須** |

> **Note**: Presigned URLs require the S3 server to validate `s3v4` signatures and honor the `Expires`
> parameter. The default TTL is 3600 seconds (1 hour), maximum 86400 seconds (24 hours).
>
> **注意**: Presigned URLにはS3サーバが`s3v4`署名を検証し、`Expires`パラメータを正しく処理する必要があります。
> デフォルトTTLは3600秒（1時間）、最大86400秒（24時間）です。

---

## 3. Required Credentials / 必要な認証情報

### 3.1 UMRS Service Account (Admin) / UMRSサービスアカウント（管理者）

UMRS needs **one service account** with the following permissions on **all buckets** it manages:

UMRSは、管理する**全バケット**に対して以下の権限を持つ**1つのサービスアカウント**が必要です：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListAllMyBuckets",
        "s3:ListBucket",
        "s3:GetBucketLocation",
        "s3:CreateBucket",
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:HeadObject",
        "s3:HeadBucket"
      ],
      "Resource": ["arn:aws:s3:::*"]
    }
  ]
}
```

**Credentials required / 必要な認証情報:**

| Item / 項目 | Description / 説明 |
|---|---|
| `access_key` | S3 access key ID / S3アクセスキーID |
| `secret_key` | S3 secret access key / S3シークレットアクセスキー |
| `endpoint_url` | S3 endpoint (e.g. `https://lens3.example.com:8443`) / S3エンドポイント |
| `bucket_location` | Region string (e.g. `us-east-1`) / リージョン文字列 |

These are configured in the s3cfg file (`~/.s3cfg`) referenced by `admin_s3_configuration_file`
in `conf_umrs.json`.

これらは`conf_umrs.json`の`admin_s3_configuration_file`で参照されるs3cfg設定ファイル（`~/.s3cfg`）に設定します。

**Example s3cfg / s3cfg設定例:**

```ini
[default]
host_base = lens3.example.com:8443
host_bucket = lens3.example.com:8443
bucket_location = us-east-1
use_https = True
check_ssl_certificate = True
access_key = <UMRS_SERVICE_ACCESS_KEY>
secret_key = <UMRS_SERVICE_SECRET_KEY>
```

### 3.2 Per-User Accounts (Optional) / ユーザー別アカウント（オプション）

For multi-user deployments, UMRS can create per-user S3 accounts with narrower permissions
scoped to specific buckets. If Lens3 supports IAM-style user creation, each user would receive:

マルチユーザー環境では、UMRSは特定バケットに限定された権限を持つユーザー別S3アカウントを作成できます。
Lens3がIAM形式のユーザー作成に対応している場合、各ユーザーには以下の権限が付与されます：

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:ListAllMyBuckets",
    "s3:ListBucket",
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject"
  ],
  "Resource": ["arn:aws:s3:::<user-specific-bucket>/*"]
}
```

> **Note for Lens3**: The current UMRS codebase uses MinIO-specific `mc admin` CLI commands for
> user/policy management. If Lens3 does not support these commands, the admin should
> **pre-create user credentials** through Lens3's own management interface and provide the s3cfg
> files to UMRS.
>
> **Lens3に関する注意**: 現在のUMRSコードベースはユーザー/ポリシー管理にMinIO固有の`mc admin`CLIコマンドを
> 使用しています。Lens3がこれらのコマンドに対応していない場合、管理者はLens3独自の管理インターフェースを
> 通じて**ユーザー認証情報を事前作成**し、s3cfgファイルをUMRSに提供してください。

---

## 4. Bucket Naming Convention / バケット命名規則

UMRS creates buckets with the following naming pattern:

UMRSは以下の命名パターンでバケットを作成します：

```
<username>.<experiment_name>.<bucket_alias>
<username>.<experiment_name>.<bucket_alias>-output
```

**Examples / 例:**

| Bucket Name / バケット名 | Purpose / 用途 |
|---|---|
| `user1.images-and-texts.bucket-3` | User data (input) / ユーザーデータ（入力） |
| `user1.images-and-texts.bucket-3-output` | User data (output) / ユーザーデータ（出力） |
| `vision-demo` | Vision model demo input / ビジョンモデルデモ入力 |
| `hmae-demo` | HMAE demo input / HMAEデモ入力 |
| `plugin-outputs` | Plugin result storage / プラグイン結果の保存 |
| `umrs-files` | Presigned URL file access / Presigned URLファイルアクセス |

Object key prefix `_staging/<run_id>/` is also used within buckets for pipeline intermediate data.

バケット内のオブジェクトキープレフィックス `_staging/<run_id>/` はパイプラインの中間データに使用されます。

**Naming rules / 命名規則:**
- All lowercase / 全て小文字
- Spaces replaced with `-` / スペースは`-`に置換
- Underscores replaced with `-` / アンダースコアは`-`に置換
- Maximum ~63 characters / 最大約63文字

---

## 5. Object Size & Types / オブジェクトサイズと種類

| Data Type / データ種別 | Typical Size / 一般的なサイズ | Format / フォーマット |
|---|---|---|
| Input images / 入力画像 | Up to ~130 MB (e.g. 8192×8192 grayscale) | PNG, TIFF, JPEG |
| Text data / テキストデータ | 1 KB – 10 MB | TXT, JSON, CSV |
| Inference results / 推論結果 | 1 KB – 50 MB | PNG, TIFF, PyTorch `.pt` |
| Embeddings / 埋め込みベクトル | 1 KB – 10 MB | NumPy `.npy`, binary |
| Model checkpoints / モデルチェックポイント | Up to ~1.3 GB | `.pth` files |

> **Multipart upload**: For large files (>100 MB), boto3 automatically uses multipart upload.
> The S3 server must support `CreateMultipartUpload`, `UploadPart`, `CompleteMultipartUpload`,
> and `AbortMultipartUpload`.
>
> **マルチパートアップロード**: 大きなファイル（100 MB超）では、boto3が自動的にマルチパートアップロードを
> 使用します。S3サーバは`CreateMultipartUpload`、`UploadPart`、`CompleteMultipartUpload`、
> `AbortMultipartUpload`に対応している必要があります。

---

## 6. Network Requirements / ネットワーク要件

| Component / コンポーネント | Accesses S3 From / S3へのアクセス元 | Notes / 備考 |
|---|---|---|
| UMRS server (`server_umrs.py`) | Application container | Main data I/O / メインデータI/O |
| UMRS client (`client_umrs.py`) | Client container | Upload/download user data / ユーザーデータのアップロード/ダウンロード |
| Vision service (BentoML) | Compute node container | Read images, write results / 画像読み込み、結果書き込み |
| Pipeline SDK | Application container | Inter-step data exchange / ステップ間データ交換 |
| End-user browser/client | External network | Via presigned URLs only / Presigned URL経由のみ |

The S3 endpoint must be reachable from all of the above. If using HTTPS, the TLS certificate must
be valid or `check_ssl_certificate` must be set to `False` in the s3cfg file.

S3エンドポイントは上記全てのコンポーネントからアクセス可能である必要があります。HTTPSを使用する場合、
TLS証明書が有効であるか、s3cfgファイルで`check_ssl_certificate`を`False`に設定する必要があります。

---

## 7. Summary Checklist for S3 Admin / S3管理者向けチェックリスト

- [ ] **Create a service account** with full bucket/object CRUD permissions
      **サービスアカウントを作成**: バケット/オブジェクトの全CRUD権限を付与
- [ ] **Provide credentials**: access_key, secret_key, endpoint URL, region/bucket_location
      **認証情報を提供**: access_key、secret_key、エンドポイントURL、リージョン/bucket_location
- [ ] **Enable path-style addressing** (not virtual-hosted-style)
      **パス形式アドレッシングを有効化**（仮想ホスト形式ではない）
- [ ] **Support S3v4 signatures** including for presigned URLs
      **S3v4署名に対応**: Presigned URLを含む
- [ ] **Allow bucket creation** by the service account
      **サービスアカウントによるバケット作成を許可**
- [ ] **Support multipart upload** for files larger than 100 MB
      **マルチパートアップロードに対応**: 100 MB超のファイル用
- [ ] **Support batch delete** (`DeleteObjects`) for pipeline staging cleanup
      **一括削除に対応** (`DeleteObjects`): パイプラインステージングのクリーンアップ用
- [ ] **Ensure network reachability** from application containers and compute node containers
      **ネットワーク到達性を確保**: アプリケーションコンテナおよび計算ノードコンテナから
- [ ] **(Optional) Pre-create per-user credentials** if Lens3 does not support programmatic
      user creation via S3 IAM API
      **（オプション）ユーザー別認証情報を事前作成**: Lens3がS3 IAM APIによるプログラム的
      ユーザー作成に非対応の場合

---

## 8. Lens3-Specific Notes / Lens3固有の注意事項

1. **User management / ユーザー管理**

   UMRS currently uses MinIO `mc admin` commands internally for creating per-user accounts and
   attaching bucket policies. Lens3 manages users through its own web UI and API. The S3 admin
   should create credentials through Lens3 and provide them to UMRS via s3cfg config files.

   UMRSは現在、ユーザーアカウント作成とバケットポリシー付与にMinIOの`mc admin`コマンドを内部的に
   使用しています。Lens3は独自のWeb UIおよびAPIでユーザーを管理します。S3管理者はLens3を通じて
   認証情報を作成し、s3cfgファイル経由でUMRSに提供してください。

2. **Bucket creation permission / バケット作成権限**

   Verify that the Lens3 credential allows the UMRS service account to create new buckets
   dynamically at runtime. If bucket creation is not permitted, all required buckets must be
   pre-created by the admin before starting UMRS.

   Lens3の認証情報がUMRSサービスアカウントに実行時の新規バケット動的作成を許可していることを確認
   してください。バケット作成が許可されていない場合、UMRS起動前に管理者が必要な全バケットを
   事前作成する必要があります。

3. **Presigned URLs / Presigned URL**

   Confirm Lens3 supports `s3v4` presigned URL generation and validation. UMRS uses presigned
   URLs for client-side direct file transfer (up to 24-hour TTL).

   Lens3が`s3v4` Presigned URLの生成と検証に対応していることを確認してください。UMRSはPresigned
   URLを使用してクライアント側の直接ファイル転送を行います（最大TTL 24時間）。

4. **Batch delete / 一括削除**

   UMRS uses `delete_objects()` (multi-object delete in a single API call) for pipeline staging
   cleanup. Confirm Lens3 supports this API.

   UMRSはパイプラインのステージングクリーンアップに`delete_objects()`（1回のAPI呼び出しで複数
   オブジェクトを削除）を使用します。Lens3がこのAPIに対応していることを確認してください。
