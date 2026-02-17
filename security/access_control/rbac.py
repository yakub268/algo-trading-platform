"""
Role-Based Access Control (RBAC) System
=======================================

Enterprise-grade access control with hierarchical roles, fine-grained permissions,
and policy-based security enforcement.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from ..vault.encryption import AdvancedEncryption

logger = logging.getLogger(__name__)

class Permission(Enum):
    """System permissions enumeration."""
    # System administration
    ADMIN_FULL = "admin.full"
    ADMIN_USERS = "admin.users"
    ADMIN_SYSTEM = "admin.system"
    ADMIN_SECURITY = "admin.security"

    # Trading operations
    TRADE_EXECUTE = "trade.execute"
    TRADE_VIEW = "trade.view"
    TRADE_MODIFY = "trade.modify"
    TRADE_CANCEL = "trade.cancel"

    # Portfolio management
    PORTFOLIO_VIEW = "portfolio.view"
    PORTFOLIO_MODIFY = "portfolio.modify"
    PORTFOLIO_RISK_LIMITS = "portfolio.risk_limits"

    # Strategy management
    STRATEGY_CREATE = "strategy.create"
    STRATEGY_MODIFY = "strategy.modify"
    STRATEGY_DELETE = "strategy.delete"
    STRATEGY_DEPLOY = "strategy.deploy"
    STRATEGY_VIEW = "strategy.view"

    # Data access
    DATA_VIEW_ALL = "data.view.all"
    DATA_VIEW_OWN = "data.view.own"
    DATA_EXPORT = "data.export"

    # Compliance and reporting
    COMPLIANCE_VIEW = "compliance.view"
    COMPLIANCE_GENERATE = "compliance.generate"
    COMPLIANCE_EXPORT = "compliance.export"

    # API access
    API_READ = "api.read"
    API_WRITE = "api.write"
    API_ADMIN = "api.admin"

@dataclass
class Role:
    """Represents a user role with permissions and metadata."""
    name: str
    permissions: Set[Permission]
    description: str
    parent_role: Optional[str] = None
    is_system_role: bool = False
    created_at: datetime = None
    modified_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.modified_at is None:
            self.modified_at = datetime.utcnow()

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        return permission in self.permissions

    def add_permission(self, permission: Permission):
        """Add a permission to the role."""
        self.permissions.add(permission)
        self.modified_at = datetime.utcnow()

    def remove_permission(self, permission: Permission):
        """Remove a permission from the role."""
        self.permissions.discard(permission)
        self.modified_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary for serialization."""
        return {
            'name': self.name,
            'permissions': [p.value for p in self.permissions],
            'description': self.description,
            'parent_role': self.parent_role,
            'is_system_role': self.is_system_role,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Create role from dictionary."""
        permissions = {Permission(p) for p in data['permissions']}
        return cls(
            name=data['name'],
            permissions=permissions,
            description=data['description'],
            parent_role=data.get('parent_role'),
            is_system_role=data.get('is_system_role', False),
            created_at=datetime.fromisoformat(data['created_at']),
            modified_at=datetime.fromisoformat(data['modified_at'])
        )

@dataclass
class UserRole:
    """Represents a user's role assignment with time constraints."""
    user_id: str
    role_name: str
    assigned_by: str
    assigned_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

    def is_expired(self) -> bool:
        """Check if role assignment has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'role_name': self.role_name,
            'assigned_by': self.assigned_by,
            'assigned_at': self.assigned_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserRole':
        """Create from dictionary."""
        return cls(
            user_id=data['user_id'],
            role_name=data['role_name'],
            assigned_by=data['assigned_by'],
            assigned_at=datetime.fromisoformat(data['assigned_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            is_active=data.get('is_active', True)
        )

class RoleBasedAccess:
    """
    Role-Based Access Control system.

    Features:
    - Hierarchical role inheritance
    - Fine-grained permission system
    - Temporary role assignments
    - Session-based access control
    - Audit logging of access decisions
    - Policy enforcement
    """

    def __init__(self, config):
        self.config = config
        self.encryption = AdvancedEncryption()
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[UserRole]] = {}
        self.access_policies: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

        # Built-in system roles
        self._create_system_roles()

    async def initialize(self):
        """Initialize RBAC system."""
        if self._initialized:
            return

        # Load encryption keys
        try:
            self.encryption.load_master_key()
        except FileNotFoundError:
            logger.info("Generating new master key for RBAC system")
            self.encryption.generate_master_key()

        # Load roles and assignments
        await self._load_roles()
        await self._load_user_roles()
        await self._load_access_policies()

        self._initialized = True
        logger.info("RBAC system initialized successfully")

    def create_role(self, name: str, description: str, permissions: List[Permission],
                   parent_role: Optional[str] = None) -> Role:
        """
        Create a new role.

        Args:
            name: Role name
            description: Role description
            permissions: List of permissions
            parent_role: Optional parent role for inheritance

        Returns:
            Role: Created role
        """
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")

        # Validate parent role exists
        if parent_role and parent_role not in self.roles:
            raise ValueError(f"Parent role '{parent_role}' does not exist")

        role = Role(
            name=name,
            description=description,
            permissions=set(permissions),
            parent_role=parent_role
        )

        self.roles[name] = role
        self._save_roles()

        logger.info(f"Created role: {name}")
        return role

    def assign_role_to_user(self, user_id: str, role_name: str, assigned_by: str,
                           expires_at: Optional[datetime] = None) -> bool:
        """
        Assign a role to a user.

        Args:
            user_id: User identifier
            role_name: Name of role to assign
            assigned_by: Who assigned the role
            expires_at: Optional expiration datetime

        Returns:
            bool: True if assignment successful
        """
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")

        # Check if user already has this role
        user_roles = self.user_roles.get(user_id, [])
        for existing_role in user_roles:
            if existing_role.role_name == role_name and existing_role.is_active:
                logger.warning(f"User {user_id} already has role {role_name}")
                return False

        # Create role assignment
        user_role = UserRole(
            user_id=user_id,
            role_name=role_name,
            assigned_by=assigned_by,
            assigned_at=datetime.utcnow(),
            expires_at=expires_at
        )

        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        self.user_roles[user_id].append(user_role)
        self._save_user_roles()

        logger.info(f"Assigned role '{role_name}' to user '{user_id}'")
        return True

    def revoke_role_from_user(self, user_id: str, role_name: str) -> bool:
        """
        Revoke a role from a user.

        Args:
            user_id: User identifier
            role_name: Name of role to revoke

        Returns:
            bool: True if revocation successful
        """
        user_roles = self.user_roles.get(user_id, [])

        for user_role in user_roles:
            if user_role.role_name == role_name and user_role.is_active:
                user_role.is_active = False
                self._save_user_roles()
                logger.info(f"Revoked role '{role_name}' from user '{user_id}'")
                return True

        logger.warning(f"User {user_id} does not have active role {role_name}")
        return False

    def check_permission(self, user_id: str, permission: Permission,
                        resource: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a user has a specific permission.

        Args:
            user_id: User identifier
            permission: Permission to check
            resource: Optional resource context
            context: Optional additional context

        Returns:
            bool: True if user has permission
        """
        # Get user's effective permissions
        user_permissions = self.get_user_permissions(user_id)

        # Check direct permission
        if permission in user_permissions:
            # Apply access policies if they exist
            if self._check_access_policies(user_id, permission, resource, context):
                logger.debug(f"Permission granted: {user_id} -> {permission.value}")
                return True

        logger.debug(f"Permission denied: {user_id} -> {permission.value}")
        return False

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all effective permissions for a user.

        Args:
            user_id: User identifier

        Returns:
            set: Set of permissions
        """
        all_permissions = set()
        user_roles = self.user_roles.get(user_id, [])

        for user_role in user_roles:
            if user_role.is_active and not user_role.is_expired():
                role_permissions = self._get_role_permissions(user_role.role_name)
                all_permissions.update(role_permissions)

        return all_permissions

    def get_user_roles(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all roles assigned to a user.

        Args:
            user_id: User identifier

        Returns:
            list: List of role assignments
        """
        user_roles = self.user_roles.get(user_id, [])
        return [
            {
                **user_role.to_dict(),
                'is_expired': user_role.is_expired(),
                'role_description': self.roles[user_role.role_name].description
            }
            for user_role in user_roles
        ]

    def list_roles(self) -> List[Dict[str, Any]]:
        """
        List all available roles.

        Returns:
            list: List of roles with metadata
        """
        return [role.to_dict() for role in self.roles.values()]

    def get_role(self, role_name: str) -> Optional[Role]:
        """
        Get a role by name.

        Args:
            role_name: Role name

        Returns:
            Role: Role object or None if not found
        """
        return self.roles.get(role_name)

    def update_role_permissions(self, role_name: str, permissions: List[Permission]) -> bool:
        """
        Update permissions for a role.

        Args:
            role_name: Role name
            permissions: New list of permissions

        Returns:
            bool: True if update successful
        """
        if role_name not in self.roles:
            return False

        role = self.roles[role_name]
        if role.is_system_role:
            raise ValueError("Cannot modify system roles")

        role.permissions = set(permissions)
        role.modified_at = datetime.utcnow()
        self._save_roles()

        logger.info(f"Updated permissions for role: {role_name}")
        return True

    def delete_role(self, role_name: str) -> bool:
        """
        Delete a role (if not in use).

        Args:
            role_name: Role name to delete

        Returns:
            bool: True if deletion successful
        """
        if role_name not in self.roles:
            return False

        role = self.roles[role_name]
        if role.is_system_role:
            raise ValueError("Cannot delete system roles")

        # Check if role is assigned to any users
        for user_roles in self.user_roles.values():
            for user_role in user_roles:
                if user_role.role_name == role_name and user_role.is_active:
                    raise ValueError(f"Cannot delete role '{role_name}': still assigned to users")

        del self.roles[role_name]
        self._save_roles()

        logger.info(f"Deleted role: {role_name}")
        return True

    def create_access_policy(self, name: str, policy: Dict[str, Any]):
        """
        Create an access policy for fine-grained control.

        Args:
            name: Policy name
            policy: Policy definition
        """
        self.access_policies[name] = policy
        self._save_access_policies()

    def _get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Get all permissions for a role including inherited ones."""
        if role_name not in self.roles:
            return set()

        role = self.roles[role_name]
        permissions = role.permissions.copy()

        # Add inherited permissions from parent role
        if role.parent_role:
            parent_permissions = self._get_role_permissions(role.parent_role)
            permissions.update(parent_permissions)

        return permissions

    def _check_access_policies(self, user_id: str, permission: Permission,
                              resource: Optional[str], context: Optional[Dict[str, Any]]) -> bool:
        """Check access policies for additional restrictions."""
        # Time-based restrictions
        if 'time_restrictions' in self.access_policies:
            policy = self.access_policies['time_restrictions']
            current_hour = datetime.utcnow().hour
            if 'allowed_hours' in policy:
                if current_hour not in policy['allowed_hours']:
                    return False

        # IP-based restrictions
        if 'ip_restrictions' in self.access_policies and context:
            policy = self.access_policies['ip_restrictions']
            client_ip = context.get('ip_address')
            if client_ip and 'allowed_ips' in policy:
                if client_ip not in policy['allowed_ips']:
                    return False

        # Resource-based restrictions
        if resource and 'resource_restrictions' in self.access_policies:
            policy = self.access_policies['resource_restrictions']
            if resource in policy.get('restricted_resources', []):
                # Check if user has override permission
                override_permission = policy.get('override_permission')
                if override_permission:
                    user_permissions = self.get_user_permissions(user_id)
                    if Permission(override_permission) not in user_permissions:
                        return False

        return True

    def _create_system_roles(self):
        """Create built-in system roles."""
        # Super Administrator - full access
        super_admin = Role(
            name="super_admin",
            description="Super Administrator with full system access",
            permissions={Permission.ADMIN_FULL},
            is_system_role=True
        )

        # System Administrator - system management
        system_admin = Role(
            name="system_admin",
            description="System Administrator",
            permissions={
                Permission.ADMIN_USERS, Permission.ADMIN_SYSTEM,
                Permission.ADMIN_SECURITY, Permission.STRATEGY_VIEW,
                Permission.PORTFOLIO_VIEW, Permission.DATA_VIEW_ALL
            },
            is_system_role=True
        )

        # Trading Manager - trading operations
        trading_manager = Role(
            name="trading_manager",
            description="Trading Manager with full trading permissions",
            permissions={
                Permission.TRADE_EXECUTE, Permission.TRADE_VIEW,
                Permission.TRADE_MODIFY, Permission.TRADE_CANCEL,
                Permission.PORTFOLIO_VIEW, Permission.PORTFOLIO_MODIFY,
                Permission.STRATEGY_CREATE, Permission.STRATEGY_MODIFY,
                Permission.STRATEGY_DEPLOY, Permission.STRATEGY_VIEW,
                Permission.DATA_VIEW_ALL
            },
            is_system_role=True
        )

        # Trader - basic trading permissions
        trader = Role(
            name="trader",
            description="Trader with limited trading permissions",
            permissions={
                Permission.TRADE_EXECUTE, Permission.TRADE_VIEW,
                Permission.PORTFOLIO_VIEW, Permission.STRATEGY_VIEW,
                Permission.DATA_VIEW_OWN
            },
            parent_role=None,
            is_system_role=True
        )

        # Analyst - read-only analysis access
        analyst = Role(
            name="analyst",
            description="Analyst with read-only access",
            permissions={
                Permission.PORTFOLIO_VIEW, Permission.STRATEGY_VIEW,
                Permission.DATA_VIEW_ALL, Permission.COMPLIANCE_VIEW,
                Permission.API_READ
            },
            is_system_role=True
        )

        # Compliance Officer - compliance and reporting
        compliance_officer = Role(
            name="compliance_officer",
            description="Compliance Officer with reporting access",
            permissions={
                Permission.COMPLIANCE_VIEW, Permission.COMPLIANCE_GENERATE,
                Permission.COMPLIANCE_EXPORT, Permission.DATA_VIEW_ALL,
                Permission.PORTFOLIO_VIEW, Permission.TRADE_VIEW
            },
            is_system_role=True
        )

        # Viewer - minimal read access
        viewer = Role(
            name="viewer",
            description="Viewer with minimal read access",
            permissions={
                Permission.DATA_VIEW_OWN, Permission.PORTFOLIO_VIEW
            },
            is_system_role=True
        )

        # Add all system roles
        system_roles = [
            super_admin, system_admin, trading_manager,
            trader, analyst, compliance_officer, viewer
        ]

        for role in system_roles:
            self.roles[role.name] = role

    async def _load_roles(self):
        """Load roles from encrypted storage."""
        roles_file = "security/access_control/roles.enc"
        if not Path(roles_file).exists():
            logger.info("No existing roles found, using system roles only")
            return

        try:
            with open(roles_file, 'r') as f:
                encrypted_data = json.load(f)

            decrypted_data = self.encryption.decrypt_data(encrypted_data)
            roles_data = json.loads(decrypted_data.decode())

            # Load non-system roles (system roles are created in __init__)
            for role_data in roles_data:
                if not role_data.get('is_system_role', False):
                    role = Role.from_dict(role_data)
                    self.roles[role.name] = role

            logger.info(f"Loaded {len([r for r in roles_data if not r.get('is_system_role')])} custom roles")

        except Exception as e:
            logger.error(f"Failed to load roles: {e}")

    async def _load_user_roles(self):
        """Load user role assignments from encrypted storage."""
        user_roles_file = "security/access_control/user_roles.enc"
        if not Path(user_roles_file).exists():
            logger.info("No existing user role assignments found")
            return

        try:
            with open(user_roles_file, 'r') as f:
                encrypted_data = json.load(f)

            decrypted_data = self.encryption.decrypt_data(encrypted_data)
            user_roles_data = json.loads(decrypted_data.decode())

            for user_id, roles_data in user_roles_data.items():
                user_roles = [UserRole.from_dict(role_data) for role_data in roles_data]
                self.user_roles[user_id] = user_roles

            logger.info(f"Loaded role assignments for {len(self.user_roles)} users")

        except Exception as e:
            logger.error(f"Failed to load user roles: {e}")

    async def _load_access_policies(self):
        """Load access policies from file."""
        policies_file = "security/access_control/policies.json"
        if not Path(policies_file).exists():
            # Create default policies
            self.access_policies = {
                'time_restrictions': {
                    'allowed_hours': list(range(6, 22))  # 6 AM to 10 PM
                },
                'resource_restrictions': {
                    'restricted_resources': [],
                    'override_permission': Permission.ADMIN_FULL.value
                }
            }
            self._save_access_policies()
            return

        try:
            with open(policies_file, 'r') as f:
                self.access_policies = json.load(f)

            logger.info("Loaded access policies")

        except Exception as e:
            logger.error(f"Failed to load access policies: {e}")

    def _save_roles(self):
        """Save roles to encrypted storage."""
        try:
            # Save only non-system roles
            custom_roles = [role.to_dict() for role in self.roles.values() if not role.is_system_role]
            roles_json = json.dumps(custom_roles, indent=2)

            encrypted_data = self.encryption.encrypt_data(roles_json)

            roles_file = "security/access_control/roles.enc"
            Path(roles_file).parent.mkdir(parents=True, exist_ok=True)

            with open(roles_file, 'w') as f:
                json.dump(encrypted_data, f, indent=2)

            logger.debug("Roles saved to encrypted storage")

        except Exception as e:
            logger.error(f"Failed to save roles: {e}")
            raise

    def _save_user_roles(self):
        """Save user role assignments to encrypted storage."""
        try:
            user_roles_data = {}
            for user_id, roles in self.user_roles.items():
                user_roles_data[user_id] = [role.to_dict() for role in roles]

            user_roles_json = json.dumps(user_roles_data, indent=2)
            encrypted_data = self.encryption.encrypt_data(user_roles_json)

            user_roles_file = "security/access_control/user_roles.enc"
            Path(user_roles_file).parent.mkdir(parents=True, exist_ok=True)

            with open(user_roles_file, 'w') as f:
                json.dump(encrypted_data, f, indent=2)

            logger.debug("User roles saved to encrypted storage")

        except Exception as e:
            logger.error(f"Failed to save user roles: {e}")
            raise

    def _save_access_policies(self):
        """Save access policies to file."""
        try:
            policies_file = "security/access_control/policies.json"
            Path(policies_file).parent.mkdir(parents=True, exist_ok=True)

            with open(policies_file, 'w') as f:
                json.dump(self.access_policies, f, indent=2)

            logger.debug("Access policies saved")

        except Exception as e:
            logger.error(f"Failed to save access policies: {e}")
            raise