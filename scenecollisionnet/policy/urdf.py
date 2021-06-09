import os
from collections import OrderedDict

import numpy as np
import torch
from lxml import etree as ET
from urdfpy import (
    URDF,
    Collision,
    Inertial,
    Joint,
    Link,
    Material,
    Transmission,
    Visual,
)
from urdfpy.utils import parse_origin


def configure_origin(value, device=None):
    """Convert a value into a 4x4 transform matrix.
    Parameters
    ----------
    value : None, (6,) float, or (4,4) float
        The value to turn into the matrix.
        If (6,), interpreted as xyzrpy coordinates.
    Returns
    -------
    matrix : (4,4) float or None
        The created matrix.
    """
    assert isinstance(
        value, torch.Tensor
    ), "Invalid type for origin, expect 4x4 torch tensor"
    assert value.shape == (4, 4)
    return value.float().to(device)


class TorchVisual(Visual):
    def __init__(
        self, geometry, name=None, origin=None, material=None, device=None
    ):
        self.device = device
        super().__init__(geometry, name, origin, material)

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, self.device)

    @classmethod
    def _from_xml(cls, node, path, device):
        kwargs = cls._parse(node, path)
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device
        return TorchVisual(**kwargs)


class TorchCollision(Collision):
    @classmethod
    def _from_xml(cls, node, path, device):
        kwargs = cls._parse(node, path)
        kwargs["origin"] = parse_origin(node)
        return TorchCollision(**kwargs)


class TorchLink(Link):
    _ELEMENTS = {
        "inertial": (Inertial, False, False),
        "visuals": (TorchVisual, False, True),
        "collisions": (TorchCollision, False, True),
    }

    def __init__(self, name, inertial, visuals, collisions, device=None):
        self.device = device
        super().__init__(name, inertial, visuals, collisions)

    @classmethod
    def _parse_simple_elements(cls, node, path, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(
                            t.__name__, cls.__name__
                        )
                    )
                v = [t._from_xml(n, path, device) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node, path, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, device):
        """Create an instance of this class from an XML node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        obj : :class:`URDFType`
            An instance of this class parsed from the node.
        """
        return cls(**cls._parse(node, path, device))


class TorchJoint(Joint):
    def __init__(
        self,
        name,
        joint_type,
        parent,
        child,
        axis=None,
        origin=None,
        limit=None,
        dynamics=None,
        safety_controller=None,
        calibration=None,
        mimic=None,
        device=None,
    ):
        self.device = device
        super().__init__(
            name,
            joint_type,
            parent,
            child,
            axis,
            origin,
            limit,
            dynamics,
            safety_controller,
            calibration,
            mimic,
        )

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, device=self.device)

    @property
    def axis(self):
        """(3,) float : The joint axis in the joint frame."""
        return self._axis

    @axis.setter
    def axis(self, value):
        if value is None:
            value = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        elif isinstance(value, torch.Tensor):
            assert value.shape == (
                3,
            ), "Invalid shape for axis, should be (3,)"
            value = value.to(self.device)
            value = value / torch.norm(value)
        else:
            value = torch.as_tensor(value).to(self.device)
            if value.shape != (3,):
                raise ValueError("Invalid shape for axis, should be (3,)")
            value = value / torch.norm(value)
        self._axis = value.float()

    @classmethod
    def _from_xml(cls, node, path, device):
        kwargs = cls._parse(node, path)
        kwargs["joint_type"] = str(node.attrib["type"])
        kwargs["parent"] = node.find("parent").attrib["link"]
        kwargs["child"] = node.find("child").attrib["link"]
        axis = node.find("axis")
        if axis is not None:
            axis = torch.tensor(np.fromstring(axis.attrib["xyz"], sep=" "))
        kwargs["axis"] = axis
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device

        return TorchJoint(**kwargs)

    def _rotation_matrices(self, angles, axis):
        """Compute rotation matrices from angle/axis representations.
        Parameters
        ----------
        angles : (n,) float
            The angles.
        axis : (3,) float
            The axis.
        Returns
        -------
        rots : (n,4,4)
            The rotation matrices
        """
        axis = axis / torch.norm(axis)
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
        M = torch.eye(4, device=self.device).repeat((len(angles), 1, 1))
        M[:, 0, 0] = cosa
        M[:, 1, 1] = cosa
        M[:, 2, 2] = cosa
        M[:, :3, :3] += (
            torch.ger(axis, axis).repeat((len(angles), 1, 1))
            * (1.0 - cosa)[:, np.newaxis, np.newaxis]
        )
        M[:, :3, :3] += (
            torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                device=self.device,
            ).repeat((len(angles), 1, 1))
            * sina[:, np.newaxis, np.newaxis]
        )
        return M

    def get_child_poses(self, cfg, n_cfgs):
        """Computes the child pose relative to a parent pose for a given set of
        configuration values.
        Parameters
        ----------
        cfg : (n,) float or None
            The configuration values for this joint. They are interpreted
            based on the joint type as follows:
            - ``fixed`` - not used.
            - ``prismatic`` - a translation along the axis in meters.
            - ``revolute`` - a rotation about the axis in radians.
            - ``continuous`` - a rotation about the axis in radians.
            - ``planar`` - Not implemented.
            - ``floating`` - Not implemented.
            If ``cfg`` is ``None``, then this just returns the joint pose.
        Returns
        -------
        poses : (n,4,4) float
            The poses of the child relative to the parent.
        """
        if cfg is None:
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type == "fixed":
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type in ["revolute", "continuous"]:
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            return torch.matmul(
                self.origin.repeat((n_cfgs, 1, 1)),
                self._rotation_matrices(cfg.float(), self.axis),
            )
        elif self.joint_type == "prismatic":
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            translation = torch.eye(4, device=self.device).repeat(
                (n_cfgs, 1, 1)
            )
            translation[:, :3, 3] = self.axis * cfg[:, np.newaxis]
            return torch.matmul(self.origin, translation)
        elif self.joint_type == "planar":
            raise NotImplementedError()
        elif self.joint_type == "floating":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid configuration")


class TorchURDF(URDF):

    _ELEMENTS = {
        "links": (TorchLink, True, True),
        "joints": (TorchJoint, False, True),
        "transmissions": (Transmission, False, True),
        "materials": (Material, False, True),
    }

    def __init__(
        self,
        name,
        links,
        joints=None,
        transmissions=None,
        materials=None,
        other_xml=None,
        device=None,
    ):
        self.device = device
        super().__init__(
            name, links, joints, transmissions, materials, other_xml
        )

    @staticmethod
    def load(file_obj, device=None):
        """Load a URDF from a file.
        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, str):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(
                    remove_comments=True, remove_blank_text=True
                )
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError("{} is not a file".format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return TorchURDF._from_xml(node, path, device)

    @classmethod
    def _parse_simple_elements(cls, node, path, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(
                            t.__name__, cls.__name__
                        )
                    )
                v = [t._from_xml(n, path, device) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node, path, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, device):
        valid_tags = set(["joint", "link", "transmission", "material"])
        kwargs = cls._parse(node, path, device)

        extra_xml_node = ET.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        data = ET.tostring(extra_xml_node)
        kwargs["other_xml"] = data
        kwargs["device"] = device
        return cls(**kwargs)

    def _process_cfgs(self, cfgs):
        """Process a list of joint configurations into a dictionary mapping joints to
        configuration values.
        This should result in a dict mapping each joint to a list of cfg values, one
        per joint.
        """
        joint_cfg = {}
        assert isinstance(
            cfgs, torch.Tensor
        ), "Incorrectly formatted config array"
        n_cfgs = len(cfgs)
        for i, j in enumerate(self.actuated_joints):
            joint_cfg[j] = cfgs[:, i]

        return joint_cfg, n_cfgs

    def link_fk(self, cfg=None, link=None, links=None, use_names=False):
        raise NotImplementedError("Not implemented")

    def link_fk_batch(self, cfgs=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.
        Returns
        -------
        fk : dict or (n,4,4) float
            A map from links to a (n,4,4) vector of homogenous transform matrices that
            position the links relative to the base link's frame
        """
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Process link set
        link_set = self.links

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            poses = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = (
                            joint.mimic.multiplier * cfg_vals
                            + joint.mimic.offset
                        )
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]
                poses = torch.matmul(
                    joint.get_child_poses(cfg_vals, n_cfgs), poses
                )

                if parent in fk:
                    poses = torch.matmul(fk[parent], poses)
                    break
            fk[lnk] = poses

        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def visual_geometry_fk_batch(self, cfgs=None):
        """Computes the poses of the URDF's visual geometries using fk.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.
        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the visual
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                fk[visual.geometry] = torch.matmul(lfk[link], visual.origin)
        return fk
