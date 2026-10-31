import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';

const source = '/source';
const workspace = '/workspace';
const variant = process.env.BUILD_VARIANT;
if (!['Debug', 'Release'].includes(variant)) throw Error('Variante Android inválida');
const excluded = new Set(['node_modules', '.git', '.expo', 'android', 'ios', '.docker-android-output']);
// El workspace es un volumen Linux dedicado; las dependencias tienen su propio volumen.
for (const entry of fs.readdirSync(workspace)) {
  if (entry !== 'node_modules') fs.rmSync(path.join(workspace, entry), { recursive: true, force: true });
}
fs.cpSync(source, workspace, {
  recursive: true,
  filter: candidate => {
    const relative = path.relative(source, candidate);
    if (!relative) return true;
    const first = relative.split(path.sep)[0];
    return !excluded.has(first) && !relative.split(path.sep).some(part => part.startsWith('.env'));
  },
});
const env = { ...process.env, EXPO_NO_DOTENV: '1', NODE_ENV: variant === 'Release' ? 'production' : 'development' };
if (variant === 'Release') {
  const base = new URL(env.PUBLIC_BASE_URL);
  if (base.protocol !== 'https:' || base.username || base.password || base.pathname !== '/' || base.search || base.hash) throw Error('Origen HTTPS inválido');
  const origin = base.origin;
  env.EXPO_PUBLIC_API_URL = origin;
  env.EXPO_PUBLIC_PUBLIC_BASE_URL = origin;
  env.EXPO_PUBLIC_WS_URL = origin.replace(/^https:/, 'wss:');
  env.EXPO_PUBLIC_HOTEL_WS_URL = `${env.EXPO_PUBLIC_WS_URL}/ws/hotel`;
}
function run(command, args, cwd = workspace) {
  const result = spawnSync(command, args, { cwd, env, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw Error(`${command} terminó con código ${result.status}`);
}
const lock = fs.readFileSync(path.join(workspace, 'package-lock.json'));
const hash = crypto.createHash('sha256').update(lock).update(fs.readFileSync(path.join(workspace, 'package.json'))).update(process.platform + process.arch + process.versions.modules).digest('hex');
const marker = path.join(workspace, 'node_modules/.asl-lock-sha256');
if (!fs.existsSync(marker) || fs.readFileSync(marker, 'utf8') !== hash) {
  run('npm', ['ci', '--include=dev', '--no-audit', '--no-fund']);
  fs.writeFileSync(marker, hash);
}
run('node', ['node_modules/expo/bin/cli', 'prebuild', '--platform', 'android', '--no-install']);
const oldKey = path.join(source, 'android/app/debug.keystore');
if (fs.existsSync(oldKey)) fs.copyFileSync(oldKey, path.join(workspace, 'android/app/debug.keystore'));
const android = path.join(workspace, 'android');
fs.chmodSync(path.join(android, 'gradlew'), 0o755);
run('./gradlew', [`:app:assemble${variant}`, '--max-workers=4', '--console=plain', '-Dorg.gradle.jvmargs=-Xmx4096m -XX:MaxMetaspaceSize=768m', '-PreactNativeArchitectures=armeabi-v7a,arm64-v8a'], android);
const lower = variant.toLowerCase();
const apk = path.join(android, `app/build/outputs/apk/${lower}/app-${lower}.apk`);
run(path.join(env.ANDROID_HOME, 'build-tools/36.0.0/apksigner'), ['verify', '--print-certs', apk]);
const tools = path.join(env.ANDROID_HOME, 'build-tools/36.0.0');
function inspect(command, args) {
  const result = spawnSync(command, args, { env, encoding: 'utf8' });
  if (result.status !== 0) throw Error(result.stderr || `Falló ${command}`);
  return result.stdout;
}
const manifest = inspect(path.join(tools, 'aapt'), ['dump', 'badging', apk]);
if (!manifest.includes("package: name='com.asaphvelazquez.ASLMobileAPP'")) throw Error('Identificador APK inesperado');
if (!manifest.includes("'arm64-v8a'") || !manifest.includes("'armeabi-v7a'")) throw Error('Falta una arquitectura ARM requerida');
if (variant === 'Release' && manifest.includes('application-debuggable')) throw Error('El APK release permite depuración');
const certificates = inspect(path.join(tools, 'apksigner'), ['verify', '--print-certs', apk]);
const digest = certificates.match(/certificate SHA-256 digest:\s*([a-f0-9]+)/i)?.[1]?.toLowerCase();
if (!digest) throw Error('No se pudo verificar certificado del APK');
if (variant === 'Release' && digest !== 'fac61745dc0903786fb9ede62a962b399f7348f0bb6f899b8332667591033b9c') {
  throw Error('El certificado no coincide con el APK testers existente; revisar el keystore antes de distribuir');
}
fs.copyFileSync(apk, `/output/app-${lower}.apk`);
