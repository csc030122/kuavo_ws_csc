/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: '开始',
      items: [
        'basic_usage/kuavo/docs/base_useage',
      ],
    },
    {
      type: 'category',
      label: '环境相关',
      items: [
        'basic_usage/kuavo/docs/kuaovo_docker_env',
        'basic_usage/kuavo/docs/control_nuc_ros1'
      ],
    },
    {
      type: 'category',
      label: '常见错误排查',
      items: [
        'basic_usage/kuavo/docs/question_and_answer',
      ],
    },
    {
      type: 'category',
      label: 'nMPC',
      items: [
        'basic_usage/kuavo-ros-control/readme'
      ],
    },
    {
      type: 'category',
      label: 'Changelog',
      items: [
        'basic_usage/kuavo_ros1_workspace/CHANGELOG'
      ],
    },
  ],
};

module.exports = sidebars;
