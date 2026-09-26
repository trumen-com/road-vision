/** Static export: the whole site is plain files (GitHub Pages / any host).
 *  NEXT_PUBLIC_BASE_PATH is set when the site lives under a sub-path, e.g. /road-vision on GitHub Pages. */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  ...(basePath ? { basePath, assetPrefix: basePath } : {}),
};
export default nextConfig;
