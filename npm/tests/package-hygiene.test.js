'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const PACKAGE_ROOT = path.resolve(__dirname, '..');

function readJson(filename) {
  return JSON.parse(fs.readFileSync(path.join(PACKAGE_ROOT, filename), 'utf8'));
}

function listJavaScriptFiles(directory, prefix = '') {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const relativePath = path.posix.join(prefix, entry.name);
    const absolutePath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      return listJavaScriptFiles(absolutePath, relativePath);
    }
    return entry.isFile() && entry.name.endsWith('.js') ? [relativePath] : [];
  });
}

function dryRunPack() {
  if (!process.env.npm_execpath) {
    throw new Error('npm_execpath is required to verify the package manifest');
  }

  const output = execFileSync(
    process.execPath,
    [process.env.npm_execpath, 'pack', '--dry-run', '--json', '--ignore-scripts'],
    { cwd: PACKAGE_ROOT, encoding: 'utf8' },
  );
  return JSON.parse(output)[0];
}

describe('npm package hygiene', () => {
  test('dependency trees are not tracked by Git', () => {
    const trackedDependencies = execFileSync(
      'git',
      ['ls-files', '--', ':(glob)**/node_modules/**'],
      { cwd: path.resolve(PACKAGE_ROOT, '..'), encoding: 'utf8' },
    );

    expect(trackedDependencies.trim()).toBe('');
  });

  test('manifest and lockfile root versions agree', () => {
    const manifest = readJson('package.json');
    const lockfile = readJson('package-lock.json');

    expect(lockfile.version).toBe(manifest.version);
    expect(lockfile.packages[''].version).toBe(manifest.version);
  });

  test('published tarball contains only runtime and package metadata files', () => {
    const packed = dryRunPack();
    const expectedPaths = [
      'LICENSE',
      'README.md',
      'index.js',
      'package.json',
      ...listJavaScriptFiles(path.join(PACKAGE_ROOT, 'src'), 'src'),
    ].sort();
    const packedPaths = packed.files.map((file) => file.path).sort();

    expect(packedPaths).toEqual(expectedPaths);
    expect(packed.bundled).toEqual([]);
  });
});
