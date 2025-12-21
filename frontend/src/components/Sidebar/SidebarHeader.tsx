'use client';

import { useBranding } from '@/components/Branding/BrandingProvider';

export default function SidebarHeader() {
  const branding = useBranding();

  return (
    <div className="p-6 border-b border-secondary-200 dark:border-secondary-700">
      <h1 className="text-lg branding-heading flex items-center">
        {branding?.use_image && branding.image_path ? (
          <img
            src={branding.image_path}
            alt={branding.short_name || branding.heading}
            className="w-6 h-6 mr-2"
          />
        ) : null}
        <span>{branding?.use_image ? (branding.short_name || branding.heading) : branding?.heading}</span>
      </h1>
      <p className="text-sm text-secondary-500 mt-1">{branding?.tagline}</p>
    </div>
  );
}
