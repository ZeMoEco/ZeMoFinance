import picker from '@ohos.file.picker'
import fs from '@ohos.file.fs'
import { fileShare, fileUri } from '@kit.CoreFileKit'

const BACKUP_DB_FILE_NAME = 'zemo_backup.db'
const SYNC_DB_FILE_NAME = 'zemo_sync.db'
const SYNC_FILE_RESULT_SEPARATOR = '\n'
const MODEL_PACKAGE_MAX_SELECT_NUMBER = 10

function normalizePath(path: string): string {
	let root = (path ?? '').toString().trim().replace(/\\/g, '/')
	while (root.endsWith('/')) root = root.substring(0, root.length - 1)
	return root
}

function joinUriPath(baseUri: string, child: string): string {
	const base = normalizePath(baseUri)
	const name = normalizePath(child).replace(/^\/+/, '')
	if (base == '') return name
	if (name == '') return base
	return `${base}/${name}`
}

function buildPolicyInfo(uri: string): fileShare.PolicyInfo {
	return {
		uri,
		operationMode: fileShare.OperationMode.READ_MODE | fileShare.OperationMode.WRITE_MODE,
	} as fileShare.PolicyInfo
}

function buildPolicyInfos(uris: string[]): Array<fileShare.PolicyInfo> {
	const policies = [] as Array<fileShare.PolicyInfo>
	const normalizedUris = [] as string[]
	for (let i = 0; i < uris.length; i++) {
		const normalizedUri = normalizePath(uris[i])
		if (normalizedUri.indexOf('://') <= 0) continue
		let exists = false
		for (let j = 0; j < normalizedUris.length; j++) {
			if (normalizedUris[j] == normalizedUri) {
				exists = true
				break
			}
		}
		if (exists) continue
		normalizedUris.push(normalizedUri)
		policies.push(buildPolicyInfo(normalizedUri))
	}
	return policies
}

function getParentUri(uri: string): string {
	const normalizedUri = normalizePath(uri)
	const marker = '://'
	const schemeIdx = normalizedUri.indexOf(marker)
	if (schemeIdx < 0) return ''
	const firstPathSlash = normalizedUri.indexOf('/', schemeIdx + marker.length)
	if (firstPathSlash < 0) return ''
	const lastSlash = normalizedUri.lastIndexOf('/')
	if (lastSlash <= firstPathSlash) return ''
	return normalizedUri.substring(0, lastSlash)
}

function pushUniqueUri(list: string[], uri: string): void {
	const normalizedUri = normalizePath(uri)
	if (normalizedUri == '') return
	for (let i = 0; i < list.length; i++) {
		if (list[i] == normalizedUri) return
	}
	list.push(normalizedUri)
}

function buildActivationCandidates(uri: string): string[] {
	const result = [] as string[]
	let current = normalizePath(uri)
	while (current != '') {
		pushUniqueUri(result, current)
		current = getParentUri(current)
	}
	return result
}

function ensureUriDirectory(uri: string): void {
	const dirUri = normalizePath(uri)
	if (dirUri == '') throw new Error('DOWNLOAD_DIR_EMPTY')
	const dirPath = new fileUri.FileUri(dirUri).path
	try {
		fs.mkdirSync(dirPath, true)
		return
	} catch (_e0) {}
	try {
		if (fs.accessSync(dirPath)) return
	} catch (_e1) {}
	throw new Error('下载目录初始化失败')
}

function ensureUriFile(uri: string): void {
	const targetUri = normalizePath(uri)
	if (targetUri == '') throw new Error('DOWNLOAD_FILE_EMPTY')
	try {
		const filePath = new fileUri.FileUri(targetUri).path
		const slashIdx = filePath.lastIndexOf('/')
		if (slashIdx > 0) {
			try { fs.mkdirSync(filePath.substring(0, slashIdx), true) } catch (_e0) {}
		}
		const file = fs.openSync(filePath, fs.OpenMode.CREATE | fs.OpenMode.READ_WRITE)
		try {
			return
		} finally {
			try { fs.closeSync(file) } catch (_e1) {}
		}
	} catch (_e1) {
		throw new Error('初始化固定同步 DB 失败')
	}
}

async function persistUriPermission(uri: string): Promise<void> {
	await persistUriPermissions([uri])
}

async function persistUriPermissions(uris: string[]): Promise<void> {
	const policies = buildPolicyInfos(uris)
	if (policies.length === 0) return
	if (!canIUse('SystemCapability.FileManagement.AppFileService.FolderAuthorization')) {
		throw new Error('当前设备不支持文件权限持久化')
	}
	try {
		await fileShare.persistPermission(policies)
		await fileShare.activatePermission(policies)
	} catch (e) {
		const msg = e != null ? (e as Error).message : ''
		throw new Error(msg != '' ? `文件权限持久化失败: ${msg}` : '文件权限持久化失败')
	}
}

export async function activatePersistedUriPermissionNative(uri: string): Promise<boolean> {
	const normalizedUri = normalizePath(uri)
	if (normalizedUri.indexOf('://') <= 0) return true
	if (!canIUse('SystemCapability.FileManagement.AppFileService.FolderAuthorization')) {
		return false
	}
	const candidates = buildActivationCandidates(normalizedUri)
	for (let i = 0; i < candidates.length; i++) {
		try {
			const policies: Array<fileShare.PolicyInfo> = [buildPolicyInfo(candidates[i])]
			await fileShare.activatePermission(policies)
			return true
		} catch (_e0) {}
	}
	return false
}

async function createDownloadRootUri(): Promise<string> {
	const options = new picker.DocumentSaveOptions()
	options.pickerMode = picker.DocumentPickerMode.DOWNLOAD

	const documentPicker = buildDocumentPicker()
	const result = await documentPicker.save(options)
	if (result == null || result.length === 0) {
		throw new Error('DOWNLOAD_PICK_EMPTY')
	}
	const uri = normalizePath(result[0])
	const activated = await activatePersistedUriPermissionNative(uri)
	if (!activated) throw new Error('下载目录权限激活失败')
	return uri
}

function safeLowerFileName(path: string): string {
	const normalizedPath = normalizePath(path)
	const slashIdx = normalizedPath.lastIndexOf('/')
	let raw = slashIdx >= 0 ? normalizedPath.substring(slashIdx + 1) : normalizedPath
	const queryIdx = raw.indexOf('?')
	if (queryIdx >= 0) raw = raw.substring(0, queryIdx)
	const hashIdx = raw.indexOf('#')
	if (hashIdx >= 0) raw = raw.substring(0, hashIdx)
	try {
		return decodeURIComponent(raw).toLowerCase()
	} catch (_e0) {
		return raw.toLowerCase()
	}
}

function resolveSyncFileSlot(path: string): string {
	const fileName = safeLowerFileName(path)
	if (fileName == BACKUP_DB_FILE_NAME) return 'backup'
	if (fileName == SYNC_DB_FILE_NAME) return 'sync'
	return ''
}

function resolveSyncFilePairUris(uris: string[]): string[] {
	let backupUri = ''
	let syncUri = ''
	for (let i = 0; i < uris.length; i++) {
		const normalizedUri = normalizePath(uris[i])
		if (normalizedUri == '') continue
		const slot = resolveSyncFileSlot(normalizedUri)
		if (slot == 'backup') {
			if (backupUri != '') throw new Error('只能选择一个 zemo_backup.db')
			backupUri = normalizedUri
			continue
		}
		if (slot == 'sync') {
			if (syncUri != '') throw new Error('只能选择一个 zemo_sync.db')
			syncUri = normalizedUri
			continue
		}
		throw new Error('请选择文件名为 zemo_backup.db 和 zemo_sync.db 的数据库文件')
	}
	if (backupUri == '' || syncUri == '') {
		throw new Error('请同时选择 zemo_backup.db 和 zemo_sync.db')
	}
	return [backupUri, syncUri]
}

function buildSyncFileResultPayload(backupUri: string, syncUri: string): string {
	const backupPath = encodeURIComponent(normalizePath(backupUri))
	const syncPath = encodeURIComponent(normalizePath(syncUri))
	return `${backupPath}${SYNC_FILE_RESULT_SEPARATOR}${syncPath}`
}

function buildDocumentPicker(): picker.DocumentViewPicker {
	return new picker.DocumentViewPicker()
}

async function pickFolderUri(): Promise<string> {
	const options = new picker.DocumentSelectOptions()
	options.selectMode = picker.DocumentSelectMode.FOLDER
	options.maxSelectNumber = 1

	const documentPicker = buildDocumentPicker()
	const result = await documentPicker.select(options)
	if (result == null || result.length === 0) {
		throw new Error('FOLDER_PICK_EMPTY')
	}
	const uri = normalizePath(result[0])
	await persistUriPermission(uri)
	return uri
}

async function activateSyncFilePair(backupUri: string, syncUri: string): Promise<void> {
	const backupActivated = await activatePersistedUriPermissionNative(backupUri)
	const syncActivated = await activatePersistedUriPermissionNative(syncUri)
	if (!backupActivated || !syncActivated) {
		throw new Error('固定同步 DB 权限激活失败')
	}
}

async function selectSyncFilePair(): Promise<string> {
	const options = new picker.DocumentSelectOptions()
	options.maxSelectNumber = 2

	const documentPicker = buildDocumentPicker()
	const result = await documentPicker.select(options)
	if (result == null || result.length === 0) {
		throw new Error('FILE_PICK_EMPTY')
	}
	const pairUris = resolveSyncFilePairUris(result)
	await persistUriPermissions(pairUris)
	return buildSyncFileResultPayload(pairUris[0], pairUris[1])
}

export async function chooseModelPackageNative(): Promise<string> {
	const options = new picker.DocumentSelectOptions()
	options.maxSelectNumber = MODEL_PACKAGE_MAX_SELECT_NUMBER

	const documentPicker = buildDocumentPicker()
	const result = await documentPicker.select(options)
	if (result == null || result.length === 0) {
		throw new Error('MODEL_PICK_EMPTY')
	}
	const uris = [] as string[]
	for (let i = 0; i < result.length; i++) {
		const uri = normalizePath(result[i])
		if (uri != '') uris.push(uri)
	}
	await persistUriPermissions(uris)
	return uris.map((uri: string): string => encodeURIComponent(uri)).join(SYNC_FILE_RESULT_SEPARATOR)
}

async function createSyncFilePair(): Promise<string> {
	const options = new picker.DocumentSaveOptions()
	options.newFileNames = [BACKUP_DB_FILE_NAME, SYNC_DB_FILE_NAME]

	const documentPicker = buildDocumentPicker()
	const result = await documentPicker.save(options)
	if (result == null || result.length === 0) {
		throw new Error('FILE_SAVE_EMPTY')
	}
	const pairUris = resolveSyncFilePairUris(result)
	await persistUriPermissions(pairUris)
	return buildSyncFileResultPayload(pairUris[0], pairUris[1])
}

export async function choosePublicDirectoryNative(): Promise<string> {
	let root = ''
	let folderError = '目录选择失败'
	try {
		root = await createDownloadRootUri()
	} catch (_e0) {
		folderError = 'DOWNLOAD 模式不可用'
	}

	if (root.length === 0) {
		try {
			root = await pickFolderUri()
		} catch (_e1) {
			throw new Error(`公共目录授权失败（${folderError}；目录选择也失败，请重新选择同步位置）`)
		}
	}

	return root
}

export async function chooseSyncFileNative(create: boolean): Promise<string> {
	if (create) {
		return await createSyncFilePair()
	}
	return await selectSyncFilePair()
}
